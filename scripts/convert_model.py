"""Convert Helsinki-NLP/opus-mt-en-es MarianMT model to CTranslate2 INT8.

Generates a standalone, lightweight model directory (<200MB RAM footprint)
containing the converted CTranslate2 model binary and pure SentencePiece models.
"""

import argparse
import logging
import sys
from pathlib import Path

import ctranslate2
import sentencepiece as spm
from ctranslate2.converters import TransformersConverter
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("convert_model")


def convert_model(
    model_name: str = "Helsinki-NLP/opus-mt-en-es",
    output_dir: str = "models/opus-mt-en-es-ct2-int8",
    quantization: str = "int8",
    force: bool = False,
) -> Path:
    """Convert HuggingFace MarianMT model to CTranslate2 with INT8 quantization."""
    out_path = Path(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and (out_path / "model.bin").exists() and not force:
        logger.info("Converted model already exists at %s. Use --force to reconvert.", out_path)
        return out_path

    logger.info("Starting model conversion for %s to %s (quantization=%s)", model_name, out_path, quantization)

    # 1. Convert MarianMT architecture weights to CTranslate2 format
    logger.info("Step 1/3: Converting model weights with CTranslate2 TransformersConverter...")
    converter = TransformersConverter(
        model_name_or_path=model_name,
        copy_files=["source.spm", "target.spm", "vocab.json", "tokenizer_config.json"],
    )
    converter.convert(
        output_dir=str(out_path),
        quantization=quantization,
        force=force or out_path.exists(),
    )

    # 2. Ensure SentencePiece tokenizer assets (source.spm, target.spm) are present
    logger.info("Step 2/3: Exporting SentencePiece tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(out_path))

    # Verify expected files exist
    expected_files = ["model.bin", "source.spm"]
    for fname in expected_files:
        p = out_path / fname
        if not p.is_file():
            raise FileNotFoundError(f"Expected converted file missing: {p}")

    # Compute model directory disk size
    total_size = sum(f.stat().st_size for f in out_path.glob("**/*") if f.is_file())
    logger.info("Model converted successfully! Directory size: %.2f MB", total_size / (1024 * 1024))

    # 3. Smoke verification test with CTranslate2 and SentencePiece
    logger.info("Step 3/3: Performing smoke verification test...")
    source_sp_file = out_path / "source.spm"
    target_sp_file = out_path / "target.spm" if (out_path / "target.spm").exists() else source_sp_file

    sp_source = spm.SentencePieceProcessor(model_file=str(source_sp_file))
    sp_target = spm.SentencePieceProcessor(model_file=str(target_sp_file))
    translator = ctranslate2.Translator(str(out_path), device="cpu", compute_type=quantization)

    sample_src = "Artificial intelligence is changing the world."
    src_tokens = sp_source.encode(sample_src, out_type=str) + ["</s>"]
    res = translator.translate_batch([src_tokens], beam_size=4, max_decoding_length=256)
    clean_hyp = [t for t in res[0].hypotheses[0] if t != "</s>"]
    pred_es = sp_target.decode(clean_hyp)

    logger.info("Smoke Test Verification:")
    logger.info("  Input (EN) : %s", sample_src)
    logger.info("  Output (ES): %s", pred_es)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert MarianMT to CTranslate2 INT8")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-es",
        help="HuggingFace model ID or local directory (default: Helsinki-NLP/opus-mt-en-es)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/opus-mt-en-es-ct2-int8",
        help="Target output directory for converted model (default: models/opus-mt-en-es-ct2-int8)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="Weight quantization type (default: int8)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    args = parser.parse_args()

    convert_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        quantization=args.quantization,
        force=args.force,
    )


if __name__ == "__main__":
    main()
