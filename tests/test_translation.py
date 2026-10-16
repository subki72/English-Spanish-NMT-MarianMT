"""Unit tests for TranslationEngine and input validation logic."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.engine import TranslationEngine
from app.schemas import sanitize_and_validate_text


class TestInputValidation:
    """Tests for input validation and sanitization (BLK-04)."""

    def test_valid_text_sanitization(self):
        result = sanitize_and_validate_text("  Artificial intelligence is great.  ")
        assert result == "Artificial intelligence is great."

    def test_multiline_valid_text(self):
        text = "First sentence.\nSecond sentence.\tThird with tab."
        result = sanitize_and_validate_text(text)
        assert result == text

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_and_validate_text("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_and_validate_text("   \n\t  ")

    def test_length_exceeded_rejected(self):
        long_text = "a" * 1001
        with pytest.raises(ValueError, match="exceeds maximum allowed length"):
            sanitize_and_validate_text(long_text, max_chars=1000)

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError, match="invalid control"):
            sanitize_and_validate_text("Hello\x00World")

    def test_control_character_rejected(self):
        with pytest.raises(ValueError, match="invalid control"):
            sanitize_and_validate_text("Hello\x07World")

    def test_escape_character_rejected(self):
        with pytest.raises(ValueError, match="invalid control"):
            sanitize_and_validate_text("Hello\x1b[31mWorld")

    def test_compound_emoji_with_zwj_accepted(self):
        text = "Family: 👨‍👩‍👧‍👦 and flag: 🏳️‍🌈"
        result = sanitize_and_validate_text(text)
        assert result == text

    def test_utf8_bom_stripped_cleanly(self):
        text = "\ufeffHello world with BOM."
        result = sanitize_and_validate_text(text)
        assert result == "Hello world with BOM."

    def test_non_string_type_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            sanitize_and_validate_text(12345)  # type: ignore


class TestTranslationEngine:
    """Tests for TranslationEngine lifecycle and translation logic."""

    def test_engine_unloaded_raises_runtime_error(self, tmp_path):
        engine = TranslationEngine(model_dir=tmp_path)
        assert not engine.is_ready
        with pytest.raises(RuntimeError, match="not ready"):
            engine.translate_batch(["Hello world"])

    def test_engine_empty_batch_returns_empty_list(self, tmp_path):
        engine = TranslationEngine(model_dir=tmp_path)
        engine.is_ready = True
        engine.translator = MagicMock()
        engine.sp_source = MagicMock()
        engine.sp_target = MagicMock()
        assert engine.translate_batch([]) == []

    def test_engine_load_missing_directory(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        engine = TranslationEngine(model_dir=missing_dir)
        with pytest.raises(FileNotFoundError, match="Model directory does not exist"):
            engine.load()
        assert not engine.is_ready
        assert engine.load_error is not None

    def test_engine_load_missing_model_bin(self, tmp_path):
        engine = TranslationEngine(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="CTranslate2 model binary not found"):
            engine.load()

    def test_engine_mock_translate_batch(self, tmp_path):
        engine = TranslationEngine(model_dir=tmp_path, default_beam_size=4, default_max_decoding_length=256)
        engine.is_ready = True

        mock_sp_source = MagicMock()
        mock_sp_source.encode.side_effect = lambda t, out_type: [" ", "Hello", " world"]
        mock_sp_target = MagicMock()
        mock_sp_target.decode.side_effect = lambda tokens: "Hola mundo"

        mock_translator = MagicMock()
        mock_res = MagicMock()
        mock_res.hypotheses = [[" ", "Hola", " mundo"]]
        mock_translator.translate_batch.return_value = [mock_res]

        engine.sp_source = mock_sp_source
        engine.sp_target = mock_sp_target
        engine.translator = mock_translator

        results = engine.translate_batch(["Hello world"])
        assert results == ["Hola mundo"]
        mock_sp_source.encode.assert_called_once_with("Hello world", out_type=str)
        mock_translator.translate_batch.assert_called_once()
        mock_sp_target.decode.assert_called_once_with([" ", "Hola", " mundo"])

    def test_real_model_inference(self):
        model_path = Path("models/opus-mt-en-es-ct2-int8")
        if not (model_path / "model.bin").is_file():
            pytest.skip("Converted model not found on disk")

        real_engine = TranslationEngine(model_dir=model_path, compute_type="int8")
        real_engine.load()
        assert real_engine.is_ready

        inputs = [
            "Artificial intelligence is changing the world.",
            "Law is mostly about logic.",
        ]
        outputs = real_engine.translate_batch(inputs, beam_size=4)
        assert len(outputs) == 2
        assert "inteligencia artificial" in outputs[0].lower()
        assert "ley" in outputs[1].lower()
