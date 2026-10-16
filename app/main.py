"""FastAPI application for English-to-Spanish MarianMT translation service.
Features:
- Dynamic batching queue running in AnyIO worker threadpool
- Liveness (/healthz) and readiness (/readyz) probes
- Prometheus metrics instrumentation (/metrics)
- OpenAPI documentation (/docs, /redoc)
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import anyio
import prometheus_client
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram

from app.config import settings
from app.engine import TranslationEngine
from app.middleware import RequestIdAndStructuredLoggingMiddleware
from app.rate_limiter import SlidingWindowRateLimiter
from app.schemas import (
    HealthResponse,
    ReadyResponse,
    TranslationItem,
    TranslationRequest,
    TranslationResponse,
)

logger = logging.getLogger("app.main")
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

# --- Prometheus Metrics ---
REQUESTS_TOTAL = Counter(
    "nmt_requests_total",
    "Total HTTP requests received",
    ["endpoint", "status"],
)
TRANSLATION_SENTENCES_TOTAL = Counter(
    "nmt_translated_sentences_total",
    "Total number of sentences translated",
)
TRANSLATION_LATENCY_HISTOGRAM = Histogram(
    "nmt_translation_latency_seconds",
    "End-to-end translation latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "nmt_batch_size_sentences",
    "Sentence count per dynamic batch",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
)
IN_FLIGHT_REQUESTS = Gauge(
    "nmt_in_flight_requests",
    "Current in-flight translation requests",
)


@dataclass
class BatchJob:
    """Internal task representing an incoming translation request awaiting dynamic batching."""

    texts: list[str]
    beam_size: int | None
    max_decoding_length: int | None
    future: asyncio.Future
    created_at: float


class DynamicBatcher:
    """Dynamic batching manager that aggregates requests within a time window
    and runs inference on an AnyIO worker thread to avoid blocking the asyncio event loop.
    """

    def __init__(
        self,
        engine: TranslationEngine,
        batch_size: int = 32,
        max_wait_ms: float = 10.0,
        max_queue_depth: int = 1000,
    ):
        self.engine = engine
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.max_queue_depth = max_queue_depth
        self._queue: asyncio.Queue[BatchJob] | None = None
        self._worker_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def queue(self) -> asyncio.Queue[BatchJob]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_depth)
        return self._queue

    @property
    def is_running(self) -> bool:
        """Check whether the batcher worker task is currently active and running."""
        return (
            self._worker_task is not None
            and not self._worker_task.done()
            and self._stop_event is not None
            and not self._stop_event.is_set()
        )

    def start(self) -> None:
        """Start the background dynamic batcher worker loop if not already running."""
        if self._worker_task is not None and not self._worker_task.done():
            logger.warning("Dynamic batcher worker is already running.")
            return
        self._stop_event = asyncio.Event()
        self._queue = asyncio.Queue(maxsize=self.max_queue_depth)
        self._worker_task = asyncio.create_task(self._batcher_loop(), name="dynamic_batcher_worker")
        logger.info(
            "Dynamic batcher worker started (batch_size=%d, max_wait_ms=%.1fms, max_queue_depth=%d)",
            self.batch_size,
            self.max_wait_ms,
            self.max_queue_depth,
        )

    async def stop(self) -> None:
        """Gracefully stop the batcher, drain pending queue, and cancel worker."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        # Drain any remaining jobs in queue and cancel their futures cleanly
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    job = self._queue.get_nowait()
                    if not job.future.done():
                        job.future.set_exception(RuntimeError("Dynamic batcher is shutting down."))
                except asyncio.QueueEmpty:
                    break
        logger.info("Dynamic batcher worker stopped.")

    async def submit(
        self,
        texts: list[str],
        beam_size: int | None = None,
        max_decoding_length: int | None = None,
    ) -> list[str]:
        """Submit sentences to the batcher queue and await execution."""
        if not self.is_running:
            raise RuntimeError("Dynamic batcher worker is not running.")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[str]] = loop.create_future()
        job = BatchJob(
            texts=texts,
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            future=future,
            created_at=time.perf_counter(),
        )
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            raise
        return await future

    async def _batcher_loop(self) -> None:
        """Continuously collect requests from queue into batches and process them."""
        while self._stop_event is not None and not self._stop_event.is_set():
            jobs: list[BatchJob] = []
            try:
                try:
                    first_job = await self.queue.get()
                except asyncio.CancelledError:
                    break

                jobs = [first_job]
                total_sentences = len(first_job.texts)
                deadline = time.monotonic() + (self.max_wait_ms / 1000.0)

                # Collect subsequent jobs within the time window up to batch_size
                while total_sentences < self.batch_size:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        next_job = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        jobs.append(next_job)
                        total_sentences += len(next_job.texts)
                    except (TimeoutError, asyncio.CancelledError):
                        break

                # Execute batch processing
                BATCH_SIZE_HISTOGRAM.observe(total_sentences)
                await self._process_batch(jobs)
            except asyncio.CancelledError:
                break
            except Exception as loop_exc:
                logger.error("Unhandled error in dynamic batcher loop: %s", loop_exc, exc_info=True)
                for j in jobs:
                    if not j.future.done():
                        j.future.set_exception(loop_exc)

    async def _process_batch(self, jobs: list[BatchJob]) -> None:
        """Group jobs by parameters and dispatch inference to AnyIO worker thread."""
        # Group jobs by (beam_size, max_decoding_length) to preserve custom parameters
        groups: dict[tuple, list[BatchJob]] = {}
        for job in jobs:
            key = (job.beam_size, job.max_decoding_length)
            groups.setdefault(key, []).append(job)

        for (beam_size, max_dec), group_jobs in groups.items():
            flat_texts: list[str] = []
            slices: list[tuple[int, int]] = []
            cur = 0
            for job in group_jobs:
                flat_texts.extend(job.texts)
                next_cur = cur + len(job.texts)
                slices.append((cur, next_cur))
                cur = next_cur

            try:
                # Protect event loop: execute CTranslate2 inference in AnyIO threadpool
                results: list[str] = await anyio.to_thread.run_sync(
                    self.engine.translate_batch,
                    flat_texts,
                    beam_size,
                    max_dec,
                )

                # Distribute results back to individual job futures
                for (start_idx, end_idx), job in zip(slices, group_jobs, strict=True):
                    if not job.future.done():
                        job.future.set_result(results[start_idx:end_idx])

                TRANSLATION_SENTENCES_TOTAL.inc(len(flat_texts))

            except Exception as exc:
                logger.error("Error executing translation batch: %s", exc, exc_info=True)
                for job in group_jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)


# Global Engine and Batcher references
engine = TranslationEngine(
    model_dir=settings.model_dir,
    sp_source_path=settings.sp_source_model,
    sp_target_path=settings.sp_target_model,
    device=settings.device,
    device_index=settings.device_index,
    compute_type=settings.compute_type,
    intra_threads=settings.intra_threads,
    inter_threads=settings.inter_threads,
    default_beam_size=settings.beam_size,
    default_max_decoding_length=settings.max_decoding_length,
)
batcher = DynamicBatcher(
    engine=engine,
    batch_size=settings.batch_size,
    max_wait_ms=settings.max_wait_ms,
    max_queue_depth=settings.max_queue_depth,
)
rate_limiter = SlidingWindowRateLimiter(
    requests_limit=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
    enabled=settings.enable_rate_limiting,
)
service_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management for startup initialization and graceful shutdown."""
    logger.info("Initializing service '%s' (v%s)...", settings.app_name, settings.app_version)
    # Attempt to load model on startup if directory exists
    try:
        if settings.model_dir.exists() and (settings.model_dir / "model.bin").exists():
            engine.load()
            logger.info("Model loaded successfully on startup.")
        else:
            logger.warning(
                "Model directory '%s' not found or incomplete. Run scripts/convert_model.py to initialize.",
                settings.model_dir,
            )
    except Exception as exc:
        logger.error("Could not load translation engine during startup: %s", exc)
        raise

    batcher.start()
    yield
    logger.info("Shutting down translation service...")
    await batcher.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-ready English-to-Spanish Neural Machine Translation (NMT) microservice. "
        "Powered by Helsinki-NLP/opus-mt-en-es converted to CTranslate2 INT8 with pure SentencePiece "
        "and dynamic request batching."
    ),
    lifespan=lifespan,
)

# Correlation ID & Structured Logging Middleware
app.add_middleware(RequestIdAndStructuredLoggingMiddleware)

# CORS middleware (Hardened: forbid credentials when wildcard origin is configured)
cors_allow_credentials = False if "*" in settings.cors_origins else True
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Request-ID",
        "Retry-After",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ],
)


# Static assets and Web UI
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse, tags=["Web UI"], include_in_schema=False)
async def serve_index():
    """Serve the interactive Neural Machine Translation web interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        REQUESTS_TOTAL.labels(endpoint="/", status="404").inc()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Web UI index not found.")
    REQUESTS_TOTAL.labels(endpoint="/", status="200").inc()
    return FileResponse(
        index_file,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness probe to confirm HTTP process is running."""
    uptime = time.time() - service_start_time
    REQUESTS_TOTAL.labels(endpoint="/healthz", status="200").inc()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        uptime_seconds=round(uptime, 2),
    )


@app.get("/readyz", response_model=ReadyResponse, tags=["Health"])
async def readiness_check(response: Response):
    """Readiness probe checking if model, tokenizers, and batcher are ready."""
    is_ready = engine.is_ready and batcher.is_running
    status_code = "200" if is_ready else "503"
    REQUESTS_TOTAL.labels(endpoint="/readyz", status=status_code).inc()

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        reasons: list[str] = []
        if not engine.is_ready:
            reasons.append(engine.load_error or "Model files not loaded")
        if not batcher.is_running:
            reasons.append("Dynamic batcher worker is not running")
        return ReadyResponse(
            ready=False,
            model_loaded=bool(engine.translator is not None),
            tokenizer_loaded=bool(engine.sp_source is not None and engine.sp_target is not None),
            batcher_running=batcher.is_running,
            details="; ".join(reasons) if reasons else "Service initializing",
        )

    return ReadyResponse(
        ready=True,
        model_loaded=True,
        tokenizer_loaded=True,
        batcher_running=True,
        details=None,
    )


@app.post("/v1/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate(payload: TranslationRequest, request: Request, response: Response):
    """Translate English sentences into Spanish using dynamic batching and CTranslate2 INT8.

    - Applies per-IP sliding window rate limiting (60 req/min default)
    - Validates string length (<= 1000 characters)
    - Rejects empty, whitespace-only, or illicit control characters
    - Protects asyncio event loop by delegating computation to AnyIO threadpool
    - Tracks generated tokens and exposes standard rate-limit headers
    """
    # 1. Rate Limiting Check
    rate_headers = await rate_limiter.check_rate_limit(request)
    for header_name, header_val in rate_headers.items():
        response.headers[header_name] = header_val

    # 2. Engine and Batcher Readiness Check
    if not engine.is_ready or not batcher.is_running:
        REQUESTS_TOTAL.labels(endpoint="/v1/translate", status="503").inc()
        detail = (
            "Dynamic batcher is not running"
            if engine.is_ready
            else f"Translation engine is not ready: {engine.load_error or 'Model not loaded'}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )

    start_time = time.perf_counter()
    IN_FLIGHT_REQUESTS.inc()

    try:
        translated_texts = await batcher.submit(
            texts=payload.text,
            beam_size=payload.beam_size,
            max_decoding_length=payload.max_decoding_length,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        TRANSLATION_LATENCY_HISTOGRAM.observe(latency_ms / 1000.0)
        REQUESTS_TOTAL.labels(endpoint="/v1/translate", status="200").inc()

        items = [
            TranslationItem(source=src, target=tgt) for src, tgt in zip(payload.text, translated_texts, strict=True)
        ]

        total_tokens = sum(engine.count_tokens(t) for t in translated_texts)
        request.state.tokens_generated = total_tokens

        model_name = settings.model_dir.name
        model_id = f"Helsinki-NLP/{model_name}" if not model_name.startswith("Helsinki-NLP") else model_name

        return TranslationResponse(
            translations=items,
            source_language="en",
            target_language="es",
            model=model_id,
            latency_ms=round(latency_ms, 2),
            tokens_generated=total_tokens,
        )

    except asyncio.QueueFull as exc:
        REQUESTS_TOTAL.labels(endpoint="/v1/translate", status="503").inc()
        logger.warning("Translation queue saturated (max_queue_depth=%d): %s", settings.max_queue_depth, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Translation queue saturated, retry later",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        REQUESTS_TOTAL.labels(endpoint="/v1/translate", status="500").inc()
        logger.error("Translation request failed: %s", exc, exc_info=True)
        request.state.error_message = str(exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal translation error occurred. Please quote your X-Request-ID for assistance.",
        ) from exc
    finally:
        IN_FLIGHT_REQUESTS.dec()


@app.get("/metrics", tags=["Observability"])
async def metrics():
    """Prometheus metrics scrape endpoint."""
    if not settings.enable_metrics:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics disabled.")
    return Response(
        content=prometheus_client.generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
