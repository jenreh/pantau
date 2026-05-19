from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pantau.audio.protocol import RecognitionResult
from pantau.config import SttConfig


@pytest.fixture
def cfg() -> SttConfig:
    return SttConfig(provider="vosk", language="de", silence_stop_s=0.4)


@pytest.fixture
def adapter(cfg: SttConfig) -> object:
    from pantau.audio.backends.vosk import VoskAdapter

    with patch.object(VoskAdapter, "__init__", return_value=None):
        instance = VoskAdapter.__new__(VoskAdapter)
        instance._cfg = cfg
        rec = MagicMock()
        rec.Reset = MagicMock()
        rec.AcceptWaveform = MagicMock(return_value=False)
        rec.PartialResult = MagicMock(return_value=json.dumps({"partial": ""}))
        rec.FinalResult = MagicMock(
            return_value=json.dumps({"text": "Licht einschalten"})
        )
        instance._rec = rec
        yield instance


def test_run_returns_final_result_on_silence(adapter: object) -> None:
    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    chunk = np.zeros((480, 1), dtype=np.int16)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.backends.vosk.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        result = adapter._run(initial_silence_timeout_s=0.4)

    assert isinstance(result, RecognitionResult)
    assert result.text == "Licht einschalten"


def test_run_returns_immediately_on_accept_waveform(adapter: object) -> None:
    adapter._rec.AcceptWaveform.return_value = True
    adapter._rec.Result = MagicMock(
        return_value=json.dumps({"text": "Wohnzimmer dimmen"})
    )

    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    chunk = np.zeros((480, 1), dtype=np.int16)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.backends.vosk.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        result = adapter._run(initial_silence_timeout_s=1.2)

    assert result.text == "Wohnzimmer dimmen"


def test_run_resets_recognizer_before_recording(adapter: object) -> None:
    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    chunk = np.zeros((480, 1), dtype=np.int16)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.backends.vosk.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        adapter._run(initial_silence_timeout_s=0.4)

    adapter._rec.Reset.assert_called_once()
