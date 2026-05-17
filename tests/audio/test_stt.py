from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pantau.config import SttConfig


@pytest.fixture
def stt_cfg() -> SttConfig:
    return SttConfig(model_size="tiny", device="cpu", language="de")


@pytest.fixture
def stt(stt_cfg: SttConfig) -> object:
    from pantau.audio.stt import GermanSTT

    with patch("pantau.audio.stt.GermanSTT.__init__") as mock_init:
        mock_init.return_value = None
        instance = GermanSTT.__new__(GermanSTT)
        instance._model = MagicMock()
        instance._vad = MagicMock()
        instance._language = "de"
        yield instance


def test_transcribe_joins_segments(stt: object) -> None:
    seg1 = MagicMock()
    seg1.text = " Hallo "
    seg2 = MagicMock()
    seg2.text = " Welt "
    stt._model.transcribe.return_value = ([seg1, seg2], MagicMock())

    result = stt._transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hallo Welt"
    stt._model.transcribe.assert_called_once()


def test_transcribe_empty_audio_returns_empty(stt: object) -> None:
    stt._model.transcribe.return_value = ([], MagicMock())
    result = stt._transcribe(np.zeros(16000, dtype=np.float32))
    assert result == ""


@pytest.mark.asyncio
async def test_record_and_transcribe_calls_both(stt: object) -> None:
    audio = np.zeros(16000, dtype=np.float32)
    stt._record = MagicMock(return_value=audio)
    stt._transcribe = MagicMock(return_value="Wohnzimmer einschalten")

    with patch("pantau.audio.stt.asyncio.to_thread", side_effect=lambda fn, *a: fn(*a)):
        result = await stt.record_and_transcribe()

    assert result == "Wohnzimmer einschalten"
    stt._record.assert_called_once()
    stt._transcribe.assert_called_once_with(audio)


def test_record_stops_on_silence(stt: object) -> None:
    stt._vad.is_speech.return_value = False

    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)

    chunk = np.zeros((512, 1), dtype=np.float32)
    fake_stream.read.return_value = (chunk, None)

    with patch("pantau.audio.stt.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        result = stt._record()

    assert isinstance(result, np.ndarray)
    assert stt._vad.is_speech.called
