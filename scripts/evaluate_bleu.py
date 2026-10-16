"""Standalone BLEU evaluation script using SacreBLEU and CTranslate2 INT8.
Evaluates translation accuracy and runtime throughput on CPU.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import ctranslate2
import sacrebleu
import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_bleu")

# Curated reference benchmark pairs from OPUS / TED talks
FALLBACK_BENCHMARK_PAIRS = [
    ("Artificial intelligence is changing the world.", "La inteligencia artificial está cambiando el mundo."),
    ("Law is mostly about logic.", "La ley se trata principalmente de lógica."),
    ("I want to share an idea with you today.", "Quiero compartir una idea con ustedes hoy."),
    (
        "Machine translation has improved dramatically over the past few years.",
        "La traducción automática ha mejorado drásticamente en los últimos años.",
    ),
    ("The global climate is experiencing rapid shifts.", "El clima global está experimentando cambios rápidos."),
    (
        "Education is the most powerful weapon which you can use to change the world.",
        "La educación es el arma más poderosa que puedes usar para cambiar el mundo.",
    ),
    (
        "Scientists have discovered new evidence of water on Mars.",
        "Los científicos han descubierto nuevas evidencias de agua en Marte.",
    ),
    (
        "Effective communication requires both empathy and active listening.",
        "La comunicación eficaz requiere tanto empatía como escucha activa.",
    ),
    (
        "Renewable energy sources are becoming more cost-effective every day.",
        "Las fuentes de energía renovable son cada día más rentables.",
    ),
    ("Thank you very much for your time and attention.", "Muchas gracias por su tiempo y atención."),
]


def load_dataset_samples(
    dataset_name: str = "Helsinki-NLP/opus-100",
    lang_pair: str = "en-es",
    split: str = "test",
    max_samples: int = 500,
    eval_file: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load evaluation samples from local file, HuggingFace datasets, or fallback benchmark."""
    if eval_file and Path(eval_file).is_file():
        logger.info("Loading evaluation dataset from local file: %s", eval_file)
        sources, targets = [], []
        p = Path(eval_file)
        if p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    sources.append(item["source"])
                    targets.append(item["target"])
        else:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        sources.append(parts[0])
                        targets.append(parts[1])
        return sources[:max_samples], targets[:max_samples]

    try:
        from datasets import load_dataset

        logger.info("Loading '%s' (%s, split='%s') from HuggingFace...", dataset_name, lang_pair, split)
        ds = load_dataset(dataset_name, lang_pair, split=split)
        sources: list[str] = []
        targets: list[str] = []
        count = 0
        for item in ds:
            trans = item.get("translation", {})
            src = trans.get("en", "").strip()
            tgt = trans.get("es", "").strip()
            if src and tgt:
                sources.append(src)
                targets.append(tgt)
                count += 1
                if max_samples and count >= max_samples:
                    break
        logger.info("Loaded %d samples from %s.", len(sources), dataset_name)
        return sources, targets
    except Exception as exc:
        logger.warning(
            "Could not load HuggingFace dataset (%s). Falling back to curated benchmark set.",
            exc,
        )
        sources = [pair[0] for pair in FALLBACK_BENCHMARK_PAIRS]
        targets = [pair[1] for pair in FALLBACK_BENCHMARK_PAIRS]
        return sources, targets


def evaluate(
    model_dir: str = "models/opus-mt-en-es-ct2-int8",
    dataset_name: str = "Helsinki-NLP/opus-100",
    lang_pair: str = "en-es",
    max_samples: int = 500,
    batch_size: int = 32,
    beam_size: int = 4,
    max_decoding_length: int = 256,
    eval_file: str | None = None,
    output_json: str | None = None,
) -> dict:
    """Run BLEU evaluation on the CTranslate2 model."""
    m_dir = Path(model_dir)
    source_sp = m_dir / "source.spm"
    target_sp = m_dir / "target.spm" if (m_dir / "target.spm").exists() else source_sp

    if not m_dir.exists() or not (m_dir / "model.bin").exists():
        raise FileNotFoundError(f"CTranslate2 model not found at {m_dir}. Run scripts/convert_model.py first.")

    logger.info("Loading CTranslate2 model from %s...", m_dir)
    sp_source = spm.SentencePieceProcessor(model_file=str(source_sp))
    sp_target = spm.SentencePieceProcessor(model_file=str(target_sp))
    translator = ctranslate2.Translator(str(m_dir), device="cpu", compute_type="int8", intra_threads=4)

    sources, references = load_dataset_samples(
        dataset_name=dataset_name,
        lang_pair=lang_pair,
        max_samples=max_samples,
        eval_file=eval_file,
    )
    logger.info("Evaluating on %d test sentences (batch_size=%d, beam_size=%d)...", len(sources), batch_size, beam_size)

    hypotheses: list[str] = []
    start_time = time.perf_counter()

    for idx in range(0, len(sources), batch_size):
        chunk = sources[idx : idx + batch_size]
        tokens_batch = []
        for text in chunk:
            toks = sp_source.encode(text, out_type=str)
            if not toks or toks[-1] != "</s>":
                toks.append("</s>")
            tokens_batch.append(toks)
        results = translator.translate_batch(
            tokens_batch,
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            return_scores=False,
        )
        for res in results:
            if res.hypotheses:
                clean_hyp = [t for t in res.hypotheses[0] if t != "</s>"]
                hyp = sp_target.decode(clean_hyp)
            else:
                hyp = ""
            hypotheses.append(hyp)

    elapsed = time.perf_counter() - start_time
    throughput = len(sources) / elapsed if elapsed > 0 else 0.0
    avg_latency_ms = (elapsed / len(sources)) * 1000.0 if sources else 0.0

    # Calculate SacreBLEU & chrF++
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    metrics = {
        "sacrebleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "num_samples": len(sources),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_sentences_per_sec": round(throughput, 2),
        "avg_latency_per_sentence_ms": round(avg_latency_ms, 2),
        "beam_size": beam_size,
        "max_decoding_length": max_decoding_length,
    }

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("SacreBLEU Score              : %.2f", metrics["sacrebleu"])
    logger.info("chrF++ Score                 : %.2f", metrics["chrf"])
    logger.info("Test Sentences Evaluated     : %d", metrics["num_samples"])
    logger.info("Total Time Elapsed           : %.2f s", metrics["elapsed_seconds"])
    logger.info("Throughput                   : %.2f sent/sec", metrics["throughput_sentences_per_sec"])
    logger.info("Avg Latency Per Sentence     : %.2f ms", metrics["avg_latency_per_sentence_ms"])
    logger.info("=" * 60)

    # Show qualitative samples
    logger.info("Qualitative Prediction Samples:")
    sample_count = min(3, len(sources))
    for i in range(sample_count):
        logger.info("  Sample #%d:", i + 1)
        logger.info("    Source   : %s", sources[i])
        logger.info("    Reference: %s", references[i])
        logger.info("    Predicted: %s", hypotheses[i])

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics written to %s", output_json)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CTranslate2 NMT model using SacreBLEU")
    parser.add_argument("--model_dir", type=str, default="models/opus-mt-en-es-ct2-int8", help="Model directory")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Helsinki-NLP/opus-100",
        help="HuggingFace dataset name (default: Helsinki-NLP/opus-100)",
    )
    parser.add_argument(
        "--lang_pair",
        type=str,
        default="en-es",
        help="Language pair code (default: en-es)",
    )
    parser.add_argument("--max_samples", type=int, default=500, help="Max test sentences to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size")
    parser.add_argument("--max_decoding_length", type=int, default=256, help="Max target tokens")
    parser.add_argument("--eval_file", type=str, default=None, help="Optional local eval file (.json or .tsv)")
    parser.add_argument("--output_json", type=str, default=None, help="Save evaluation metrics to JSON")
    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        dataset_name=args.dataset_name,
        lang_pair=args.lang_pair,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        max_decoding_length=args.max_decoding_length,
        eval_file=args.eval_file,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
