"""Comprehensive API test suite for English-Spanish NMT MarianMT Service.
Covers health probes, input sanitization, dynamic batching, and error paths.
"""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest
from httpx import ASGITransport

from app.main import app, batcher, engine, rate_limiter


@pytest.fixture(autouse=True)
def reset_engine_state():
    """Ensure engine and rate limiter state is clean before each test."""
    original_ready = engine.is_ready
    original_translator = engine.translator
    original_sp_source = engine.sp_source
    original_sp_target = engine.sp_target
    original_error = engine.load_error
    original_translate_batch = engine.translate_batch
    original_rl_limit = rate_limiter.requests_limit
    original_rl_window = rate_limiter.window_seconds
    original_rl_enabled = rate_limiter.enabled
    rate_limiter.reset()

    yield

    engine.is_ready = original_ready
    engine.translator = original_translator
    engine.sp_source = original_sp_source
    engine.sp_target = original_sp_target
    engine.load_error = original_error
    engine.translate_batch = original_translate_batch
    rate_limiter.requests_limit = original_rl_limit
    rate_limiter.window_seconds = original_rl_window
    rate_limiter.enabled = original_rl_enabled
    rate_limiter.reset()


@pytest.fixture
def mock_ready_engine():
    """Fixture that configures engine to be ready with mock translation."""
    engine.is_ready = True
    engine.load_error = None

    def fake_translate_batch(texts, beam_size=None, max_decoding_length=None):
        lookup = {
            "Artificial intelligence is changing the world.": "La inteligencia artificial está cambiando el mundo.",
            "Law is mostly about logic.": "La ley se trata principalmente de lógica.",
            "Hello world.": "Hola mundo.",
            "Good morning.": "Buenos días.",
        }
        return [lookup.get(t, f"Traducción de: {t}") for t in texts]

    mock_translator = MagicMock()
    engine.translate_batch = MagicMock(side_effect=fake_translate_batch)
    engine.translator = mock_translator
    engine.sp_source = MagicMock()
    engine.sp_target = MagicMock()
    return engine


@pytest.mark.asyncio
async def test_healthz_endpoint():
    """Verify /healthz returns 200 and liveness metadata."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_readyz_not_ready():
    """Verify /readyz returns 503 when model is not loaded."""
    engine.is_ready = False
    engine.load_error = "Model files missing"
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["ready"] is False
        assert "Model files missing" in data["details"]


@pytest.mark.asyncio
async def test_readyz_ready(mock_ready_engine):
    """Verify /readyz returns 200 when model and batcher are ready."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/readyz")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ready"] is True
            assert data["model_loaded"] is True
            assert data["tokenizer_loaded"] is True
            assert data["batcher_running"] is True
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_readyz_batcher_not_running_returns_503(mock_ready_engine):
    """Verify /readyz returns 503 when model is loaded but batcher is not running."""
    await batcher.stop()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["ready"] is False
        assert data["batcher_running"] is False
        assert "Dynamic batcher" in data["details"]


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Verify /metrics exports Prometheus formatted metrics."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "nmt_requests_total" in resp.text


@pytest.mark.asyncio
async def test_translate_unready_engine_returns_503():
    """Verify /v1/translate returns 503 when engine is not ready."""
    engine.is_ready = False
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/translate",
            json={"text": "Hello world."},
        )
        assert resp.status_code == 503
        assert "not ready" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_translate_unready_batcher_returns_503(mock_ready_engine):
    """Verify /v1/translate returns 503 when batcher is not running."""
    await batcher.stop()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/translate",
            json={"text": "Hello world."},
        )
        assert resp.status_code == 503
        assert "Dynamic batcher is not running" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_translate_validation_empty_string(mock_ready_engine):
    """Verify empty string is rejected with 422."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/translate", json={"text": ""})
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_translate_validation_whitespace(mock_ready_engine):
    """Verify whitespace-only string is rejected with 422."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/translate", json={"text": "    \t\n  "})
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_translate_validation_exceeds_max_chars(mock_ready_engine):
    """Verify text exceeding 1000 characters is rejected with 422."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/translate", json={"text": "A" * 1001})
        assert resp.status_code == 422
        assert "exceeds maximum allowed length" in resp.text


@pytest.mark.asyncio
async def test_translate_validation_control_characters(mock_ready_engine):
    """Verify text with illicit control characters is rejected with 422."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/translate", json={"text": "Hello\x00World"})
        assert resp.status_code == 422
        assert "control or unprintable" in resp.text


@pytest.mark.asyncio
async def test_translate_validation_empty_list(mock_ready_engine):
    """Verify empty list is rejected with 422."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/translate", json={"text": []})
        assert resp.status_code == 422
        assert "at least one sentence" in resp.text


@pytest.mark.asyncio
async def test_translate_validation_list_with_invalid_element(mock_ready_engine):
    """Verify list with an invalid sentence (e.g. whitespace) is rejected with 422."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/translate", json={"text": ["Valid sentence.", "   "]})
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_translate_validation_list_too_many_items(mock_ready_engine):
    """Verify batch exceeding 64 items is rejected with 422."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/translate", json={"text": ["Sentence"] * 65})
        assert resp.status_code == 422
        assert "exceeds maximum allowed items" in resp.text


@pytest.mark.asyncio
async def test_translate_single_sentence(mock_ready_engine):
    """Verify translating a single sentence."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={"text": "Artificial intelligence is changing the world."},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["translations"]) == 1
            assert data["translations"][0]["source"] == "Artificial intelligence is changing the world."
            assert data["translations"][0]["target"] == "La inteligencia artificial está cambiando el mundo."
            assert data["source_language"] == "en"
            assert data["target_language"] == "es"
            assert data["latency_ms"] >= 0
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_translate_batch_list(mock_ready_engine):
    """Verify translating a list of sentences."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={
                    "text": [
                        "Law is mostly about logic.",
                        "Good morning.",
                    ]
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["translations"]) == 2
            assert data["translations"][0]["target"] == "La ley se trata principalmente de lógica."
            assert data["translations"][1]["target"] == "Buenos días."
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_concurrent_dynamic_batching(mock_ready_engine):
    """Verify concurrent requests are dynamically batched and properly resolved without mixing."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            sentences = [
                "Hello world.",
                "Good morning.",
                "Artificial intelligence is changing the world.",
                "Law is mostly about logic.",
            ]

            async def send_req(sent: str):
                return await client.post("/v1/translate", json={"text": sent})

            # Fire concurrent requests
            responses = await asyncio.gather(*(send_req(s) for s in sentences))

            for resp, src in zip(responses, sentences, strict=True):
                assert resp.status_code == 200
                data = resp.json()
                assert len(data["translations"]) == 1
                assert data["translations"][0]["source"] == src
                assert len(data["translations"][0]["target"]) > 0
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_translate_compound_emoji_accepted(mock_ready_engine):
    """Verify text with compound emojis (e.g. ZWJ) is accepted and processed."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={"text": "Family: 👨‍👩‍👧‍👦 and flag: 🏳️‍🌈"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["translations"]) == 1
            assert "👨‍👩‍👧‍👦" in data["translations"][0]["source"]
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_translate_utf8_bom_stripped(mock_ready_engine):
    """Verify text with leading UTF-8 Byte Order Mark is cleanly stripped and translated."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={"text": "\ufeffHello world."},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["translations"][0]["source"] == "Hello world."
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_translate_custom_parameters(mock_ready_engine):
    """Verify custom beam_size and max_decoding_length overrides are accepted and executed."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={
                    "text": "Hello world.",
                    "beam_size": 2,
                    "max_decoding_length": 64,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["translations"]) == 1
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_translate_engine_failure_returns_500(mock_ready_engine):
    """Verify engine failure in dynamic batcher worker returns 500 and fails future cleanly."""
    mock_ready_engine.translate_batch.side_effect = RuntimeError("Inference hardware fault")
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={"text": "Hello world."},
            )
            assert resp.status_code == 500
            assert "Translation processing error" in resp.json()["detail"]
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_real_model_api_end_to_end():
    """Verify end-to-end translation through FastAPI with real CTranslate2 weights if available."""
    from pathlib import Path

    model_path = Path("models/opus-mt-en-es-ct2-int8")
    if not (model_path / "model.bin").is_file():
        pytest.skip("Converted model not found on disk")

    engine.load()
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={"text": "Artificial intelligence is changing the world."},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "inteligencia artificial" in data["translations"][0]["target"].lower()
            assert data["latency_ms"] < 200  # P50 < 35ms expected, generous margin for test
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_ui_root_endpoint_returns_html():
    """Verify GET / serves the interactive MarianMT Web UI with text/html content type and Cache-Control headers."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "no-cache" in resp.headers.get("cache-control", "")

        html = resp.text
        # Verify brand & layout components
        assert "MarianMT" in html
        assert "opus-mt-en-es" in html

        # Verify dual-pane editor elements
        assert 'id="sourceText"' in html
        assert 'id="targetText"' in html

        # Verify control & action buttons
        assert 'id="clearBtn"' in html
        assert 'id="copyBtn"' in html
        assert 'id="translateBtn"' in html
        assert 'id="beamSizeSelect"' in html
        assert 'id="swapDirectionBtn"' in html

        # Verify metrics & telemetry pills
        assert 'id="charCounter"' in html
        assert 'id="latencyBadge"' in html
        assert 'id="tokenPill"' in html
        assert 'id="healthIndicator"' in html
        assert 'id="footerHealthDot"' in html
        assert 'id="footerHealthText"' in html
        assert 'id="toast"' in html

        # Verify crisp HD vector SVGs are present
        assert "<svg" in html


@pytest.mark.asyncio
async def test_ui_static_assets_served():
    """Verify static assets (/static/styles.css and /static/app.js) are cleanly served and self-contained."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Stylesheet
        css_resp = await client.get("/static/styles.css")
        assert css_resp.status_code == 200
        assert "text/css" in css_resp.headers["content-type"]
        assert "bg-primary" in css_resp.text
        # Verify self-contained offline layout rules
        assert ".hidden" in css_resp.text
        assert ".flex" in css_resp.text
        assert ".flex-col" in css_resp.text
        assert ".btn-swap" in css_resp.text
        assert ".toast" in css_resp.text

        # JavaScript App
        js_resp = await client.get("/static/app.js")
        assert js_resp.status_code == 200
        assert "javascript" in js_resp.headers["content-type"]
        assert "triggerTranslation" in js_resp.text
        assert "swapDirectionBtn" in js_resp.text
        assert "footerHealthDot" in js_resp.text
        assert "showToast" in js_resp.text


@pytest.mark.asyncio
async def test_ui_index_not_found_returns_404(monkeypatch, tmp_path):
    """Verify GET / returns 404 if the index HTML file is missing."""
    empty_dir = tmp_path / "empty_static"
    empty_dir.mkdir()
    monkeypatch.setattr("app.main.STATIC_DIR", empty_dir)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 404
        assert "Web UI index not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_ui_sample_pills_have_distinct_svgs():
    """Verify sample text pills have unique SVGs and data-text attributes."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        html = resp.text

        # Verify semantic sample pills
        assert "AI Revolution" in html
        assert "Law &amp; Logic" in html or "Law & Logic" in html
        assert "Dynamic Batching" in html
        assert "Observability" in html
        assert 'data-text="Artificial intelligence is changing the world."' in html
        assert 'data-text="Law is mostly about logic."' in html


@pytest.mark.asyncio
async def test_translate_multiline_text(mock_ready_engine):
    """Verify translating multiline text with newlines through the API."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            multiline = "Hello world.\nLaw is mostly about logic."
            resp = await client.post("/v1/translate", json={"text": multiline})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["translations"]) == 1
            assert data["translations"][0]["source"] == multiline
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_request_id_generated_automatically():
    """Verify X-Request-ID header is generated and returned when not provided by client."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) >= 16


@pytest.mark.asyncio
async def test_request_id_preserves_custom_header():
    """Verify X-Request-ID passed by client is preserved and echoed back."""
    custom_id = "trace-uuid-abc-12345"
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/healthz", headers={"X-Request-ID": custom_id})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == custom_id


@pytest.mark.asyncio
async def test_rate_limit_headers_on_successful_translation(mock_ready_engine):
    """Verify translation endpoint returns standard RFC 6585 rate limit headers on success."""
    batcher.start()
    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/translate",
                json={"text": "Hello world."},
                headers={"X-Forwarded-For": "203.0.113.1"},
            )
            assert resp.status_code == 200
            assert "X-RateLimit-Limit" in resp.headers
            assert "X-RateLimit-Remaining" in resp.headers
            assert "X-RateLimit-Reset" in resp.headers
            assert resp.headers["X-RateLimit-Limit"] == "60"
            assert int(resp.headers["X-RateLimit-Remaining"]) == 59
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_rate_limit_exceeded_returns_429(mock_ready_engine):
    """Verify 429 Too Many Requests is returned with standard headers when quota is exceeded."""
    batcher.start()
    rate_limiter.requests_limit = 2
    rate_limiter.window_seconds = 60
    client_ip = "198.51.100.42"

    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # First 2 requests succeed
            for i in range(2):
                resp = await client.post(
                    "/v1/translate",
                    json={"text": f"Hello world {i}."},
                    headers={"X-Forwarded-For": client_ip},
                )
                assert resp.status_code == 200, f"Request {i} failed unexpectedly"

            # 3rd request should be throttled (429)
            resp_429 = await client.post(
                "/v1/translate",
                json={"text": "Hello world throttled."},
                headers={"X-Forwarded-For": client_ip},
            )
            assert resp_429.status_code == 429
            data = resp_429.json()
            assert "Rate limit exceeded" in data["detail"]
            assert "Retry-After" in resp_429.headers
            assert int(resp_429.headers["Retry-After"]) > 0
            assert resp_429.headers["X-RateLimit-Remaining"] == "0"
            assert resp_429.headers["X-RateLimit-Limit"] == "2"
            assert "X-Request-ID" in resp_429.headers
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_rate_limit_ip_isolation(mock_ready_engine):
    """Verify rate limit is enforced per IP and does not throttle other client IPs."""
    batcher.start()
    rate_limiter.requests_limit = 1
    rate_limiter.window_seconds = 60

    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Client A exhausts quota
            resp_a1 = await client.post(
                "/v1/translate",
                json={"text": "Hello world."},
                headers={"X-Forwarded-For": "10.0.0.1"},
            )
            assert resp_a1.status_code == 200

            resp_a2 = await client.post(
                "/v1/translate",
                json={"text": "Hello world again."},
                headers={"X-Forwarded-For": "10.0.0.1"},
            )
            assert resp_a2.status_code == 429

            # Client B is still fresh and succeeds
            resp_b1 = await client.post(
                "/v1/translate",
                json={"text": "Hello world from B."},
                headers={"X-Forwarded-For": "10.0.0.2"},
            )
            assert resp_b1.status_code == 200
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_structured_logging_translation_success(mock_ready_engine, caplog):
    """Verify structured JSON logging outputs request_id, client_ip, latency_ms, tokens_generated, status_code."""
    import logging

    batcher.start()
    caplog.clear()

    try:
        with caplog.at_level(logging.INFO, logger="nmt.structured"):
            async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                req_id = "test-log-uuid-999"
                client_ip = "192.0.2.77"
                resp = await client.post(
                    "/v1/translate",
                    json={"text": "Law is mostly about logic."},
                    headers={"X-Request-ID": req_id, "X-Forwarded-For": client_ip},
                )
                assert resp.status_code == 200

        # Verify structured JSON log was emitted
        json_logs = []
        for record in caplog.records:
            if record.name == "nmt.structured":
                try:
                    json_logs.append(json.loads(record.getMessage()))
                except Exception:
                    pass

        assert len(json_logs) >= 1
        matching = [log for log in json_logs if log.get("request_id") == req_id]
        assert len(matching) >= 1
        entry = matching[0]

        assert entry["request_id"] == req_id
        assert entry["client_ip"] == client_ip
        assert entry["status_code"] == 200
        assert entry["latency_ms"] >= 0
        assert entry["tokens_generated"] >= 1
        assert entry["path"] == "/v1/translate"
        assert entry["method"] == "POST"

    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_structured_logging_on_validation_error(caplog):
    """Verify structured JSON logging outputs status_code 422 and tokens_generated 0 on validation error."""
    import logging

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nmt.structured"):
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            req_id = "error-trace-111"
            resp = await client.post(
                "/v1/translate",
                json={"text": ""},
                headers={"X-Request-ID": req_id, "X-Forwarded-For": "192.0.2.88"},
            )
            assert resp.status_code == 422

    json_logs = []
    for record in caplog.records:
        if record.name == "nmt.structured":
            try:
                json_logs.append(json.loads(record.getMessage()))
            except Exception:
                pass

    matching = [log for log in json_logs if log.get("request_id") == req_id]
    assert len(matching) >= 1
    entry = matching[0]
    assert entry["request_id"] == req_id
    assert entry["status_code"] == 422
    assert entry["tokens_generated"] == 0
    assert entry["client_ip"] == "192.0.2.88"
