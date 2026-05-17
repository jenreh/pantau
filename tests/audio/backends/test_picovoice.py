from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pantau.audio.protocol import RecognitionResult
from pantau.config import SttConfig


@pytest.fixture
def cfg() -> SttConfig:
    return SttConfig(
        provider="picovoice", language="de", silence_stop_s=0.4, api_key="test-key"
    )


@pytest.fixture
def adapter(cfg: SttConfig) -> object:
    from pantau.audio.backends.picovoice import PicovoiceCheetahAdapter

    with patch.object(PicovoiceCheetahAdapter, "__init__", return_value=None):
        instance = PicovoiceCheetahAdapter.__new__(PicovoiceCheetahAdapter)
        instance._cfg = cfg
        cheetah = MagicMock()
        cheetah.frame_length = 512
        cheetah.process = MagicMock(return_value=("", False))
        cheetah.flush = MagicMock(return_value="")
        instance._cheetah = cheetah
        yield instance


def test_run_returns_empty_on_initial_silence(adapter: object) -> None:
    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    chunk = np.zeros((512, 1), dtype=np.int16)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.backends.picovoice.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        result = adapter._run(initial_silence_timeout_s=0.1)

    assert isinstance(result, RecognitionResult)
    assert result.text == ""


def test_run_returns_transcript_on_endpoint(adapter: object) -> None:
    call_count = 0

    def process_side_effect(chunk: np.ndarray) -> tuple[str, bool]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ("Licht", False)
        if call_count == 2:
            return (" an", True)
        return ("", False)

    adapter._cheetah.process.side_effect = process_side_effect
    adapter._cheetah.flush.return_value = ""

    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    chunk = np.zeros((512, 1), dtype=np.int16)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.backends.picovoice.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        result = adapter._run(initial_silence_timeout_s=6.0)

    assert result.text == "Licht an"


def test_run_appends_flush_on_endpoint(adapter: object) -> None:
    adapter._cheetah.process.return_value = ("Hallo", True)
    adapter._cheetah.flush.return_value = " Welt"

    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    chunk = np.zeros((512, 1), dtype=np.int16)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.backends.picovoice.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        result = adapter._run(initial_silence_timeout_s=6.0)

    assert result.text == "Hallo Welt"
