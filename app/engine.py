"""CTranslate2 INT8 translation engine wrapper using pure SentencePiece.
Eliminates PyTorch runtime bloat and addresses BLK-02 & BLK-05.
"""

import logging
from pathlib import Path

import ctranslate2
import sentencepiece as spm

logger = logging.getLogger(__name__)


class TranslationEngine:
    """Wrapper around CTranslate2 Translator and SentencePiece processors."""

    def __init__(
        self,
        model_dir: Path,
        sp_source_path: Path | None = None,
        sp_target_path: Path | None = None,
        device: str = "cpu",
        device_index: int = 0,
        compute_type: str = "int8",
        intra_threads: int = 4,
        inter_threads: int = 1,
        default_beam_size: int = 4,
        default_max_decoding_length: int = 256,
    ):
        self.model_dir = Path(model_dir)
        self.sp_source_path = Path(sp_source_path) if sp_source_path else None
        self.sp_target_path = Path(sp_target_path) if sp_target_path else None
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        self.default_beam_size = default_beam_size
        self.default_max_decoding_length = default_max_decoding_length

        self.translator: ctranslate2.Translator | None = None
        self.sp_source: spm.SentencePieceProcessor | None = None
        self.sp_target: spm.SentencePieceProcessor | None = None
        self.is_ready: bool = False
        self.load_error: str | None = None

    def _resolve_sp_models(self) -> tuple[Path, Path]:
        """Resolve paths for source and target SentencePiece model files."""
        # 1. Source SentencePiece resolution
        source_path = self.sp_source_path
        if not source_path:
            candidates = [
                self.model_dir / "source.spm",
                self.model_dir / "spiece.model",
                self.model_dir / "source.model",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    source_path = candidate
                    break

        if not source_path or not source_path.is_file():
            raise FileNotFoundError(
                f"Source SentencePiece model not found in {self.model_dir}. "
                f"Expected 'source.spm' or explicit path via SP_SOURCE_MODEL."
            )

        # 2. Target SentencePiece resolution (MarianMT uses target.spm, fallback to source if shared)
        target_path = self.sp_target_path
        if not target_path:
            candidates = [
                self.model_dir / "target.spm",
                self.model_dir / "target.model",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    target_path = candidate
                    break

        if not target_path or not target_path.is_file():
            logger.info("Target SentencePiece model not found separately; sharing source model.")
            target_path = source_path

        return source_path, target_path

    def load(self) -> None:
        """Load CTranslate2 model and SentencePiece tokenizers into memory."""
        try:
            logger.info(
                "Initializing TranslationEngine from %s (device=%s, compute=%s)",
                self.model_dir,
                self.device,
                self.compute_type,
            )

            if not self.model_dir.is_dir():
                raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

            model_bin = self.model_dir / "model.bin"
            if not model_bin.is_file():
                raise FileNotFoundError(f"CTranslate2 model binary not found at {model_bin}")

            source_sp_file, target_sp_file = self._resolve_sp_models()

            logger.info("Loading SentencePiece source model: %s", source_sp_file)
            self.sp_source = spm.SentencePieceProcessor(model_file=str(source_sp_file))

            logger.info("Loading SentencePiece target model: %s", target_sp_file)
            self.sp_target = spm.SentencePieceProcessor(model_file=str(target_sp_file))

            logger.info("Loading CTranslate2 Translator...")
            self.translator = ctranslate2.Translator(
                model_path=str(self.model_dir),
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                intra_threads=self.intra_threads,
                inter_threads=self.inter_threads,
            )

            self.is_ready = True
            self.load_error = None
            logger.info("TranslationEngine loaded successfully and ready for inference.")

        except Exception as exc:
            self.is_ready = False
            self.load_error = str(exc)
            logger.error("Failed to load TranslationEngine: %s", exc, exc_info=True)
            raise

    def translate_batch(
        self,
        texts: list[str],
        beam_size: int | None = None,
        max_decoding_length: int | None = None,
    ) -> list[str]:
        """Translate a batch of sentences using SentencePiece and CTranslate2.

        Parameters
        ----------
        texts : List[str]
            Pre-sanitized input sentences.
        beam_size : Optional[int]
            Beam size override.
        max_decoding_length : Optional[int]
            Max generated token length.

        Returns
        -------
        List[str]
            Translated sentences.
        """
        if not self.is_ready or not self.translator or not self.sp_source or not self.sp_target:
            raise RuntimeError(f"Engine is not ready for translation: {self.load_error or 'Not loaded'}")

        if not texts:
            return []

        active_beam_size = beam_size if beam_size is not None else self.default_beam_size
        active_max_decoding_length = (
            max_decoding_length if max_decoding_length is not None else self.default_max_decoding_length
        )

        # 1. Tokenize directly via SentencePiece with MarianMT EOS termination (BLK-05)
        tokens_batch: list[list[str]] = []
        for t in texts:
            toks = self.sp_source.encode(t, out_type=str)
            if not toks or toks[-1] != "</s>":
                toks.append("</s>")
            tokens_batch.append(toks)

        # 2. CTranslate2 INT8 batched inference (releases Python GIL)
        results = self.translator.translate_batch(
            tokens_batch,
            beam_size=active_beam_size,
            max_decoding_length=active_max_decoding_length,
            return_scores=False,
            replace_unknowns=True,
        )

        # 3. Detokenize output tokens to target language string
        translations: list[str] = []
        for result in results:
            if result.hypotheses and len(result.hypotheses) > 0:
                clean_hyp = [token for token in result.hypotheses[0] if token != "</s>"]
                detok = self.sp_target.decode(clean_hyp)
                translations.append(detok)
            else:
                translations.append("")

        return translations

    def count_tokens(self, text: str) -> int:
        """Count generated subwords or fallback to whitespace word count.
        Safe against MagicMock objects in tests.
        """
        if self.sp_target is not None:
            try:
                tokens = self.sp_target.encode(text, out_type=str)
                if isinstance(tokens, list):
                    return len(tokens)
            except Exception:
                pass
        return len(text.split())
