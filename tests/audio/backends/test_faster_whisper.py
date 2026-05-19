from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pantau.config import SttConfig


@pytest.fixture
def cfg() -> SttConfig:
    return SttConfig(model_size="tiny", device="cpu", language="de", silence_stop_s=0.4)


@pytest.fixture
def adapter(cfg: SttConfig) -> object:
    from pantau.audio.backends.faster_whisper import FasterWhisperAdapter

    with patch.object(FasterWhisperAdapter, "__init__", return_value=None):
        instance = FasterWhisperAdapter.__new__(FasterWhisperAdapter)
        instance._model = MagicMock()
        instance._vad = MagicMock()
        instance._cfg = cfg
        yield instance


def test_transcribe_joins_segments(adapter: object) -> None:
    seg1 = MagicMock()
    seg1.text = " Hallo "
    seg2 = MagicMock()
    seg2.text = " Welt "
    adapter._model.transcribe.return_value = ([seg1, seg2], MagicMock())

    result = adapter._transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hallo Welt"


def test_transcribe_empty_returns_empty(adapter: object) -> None:
    adapter._model.transcribe.return_value = ([], MagicMock())
    result = adapter._transcribe(np.zeros(16000, dtype=np.float32))
    assert result == ""


def test_transcribe_uses_config_language(adapter: object) -> None:
    adapter._model.transcribe.return_value = ([], MagicMock())
    adapter._transcribe(np.zeros(16000, dtype=np.float32))

    call_kwargs = adapter._model.transcribe.call_args.kwargs
    assert call_kwargs.get("language") == "de"


def test_record_uses_silence_stop_s_from_config(adapter: object) -> None:
    adapter._vad.is_speech.return_value = False

    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    chunk = np.zeros((512, 1), dtype=np.float32)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.backends._batch_base.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        result = adapter._record(initial_silence_timeout_s=0.4)

    assert isinstance(result, np.ndarray)
