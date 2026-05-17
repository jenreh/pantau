from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pantau.config import TtsConfig


@pytest.fixture
def tts_cfg() -> TtsConfig:
    return TtsConfig(model="models/de_DE-thorsten-high.onnx", speak_rate=1.0)


@pytest.fixture
def tts(tts_cfg: TtsConfig) -> object:
    from piper.config import SynthesisConfig

    from pantau.audio.tts import PiperTTS

    with patch("pantau.audio.tts.PiperTTS.__init__") as mock_init:
        mock_init.return_value = None
        instance = PiperTTS.__new__(PiperTTS)
        instance._voice = MagicMock()
        instance._syn_config = SynthesisConfig(length_scale=1.0 / tts_cfg.speak_rate)
        yield instance


@pytest.mark.asyncio
async def test_speak_calls_synthesize_and_play(tts: object) -> None:
    calls: list[str] = []

    def fake_synth_and_play(text: str) -> None:
        calls.append(text)

    tts._synthesize_and_play = fake_synth_and_play

    with patch("pantau.audio.tts.asyncio.to_thread", side_effect=lambda fn, *a: fn(*a)):
        await tts.speak("Hallo Welt")

    assert calls == ["Hallo Welt"]


def test_synthesize_and_play_invokes_voice(tts: object) -> None:
    import wave

    import numpy as np

    sample_rate = 22050
    samples = np.zeros(sample_rate, dtype=np.int16)

    def fake_synthesize_wav(
        text: str, wav_file: wave.Wave_write, **kwargs: object
    ) -> None:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())

    tts._voice.synthesize_wav.side_effect = fake_synthesize_wav

    with (
        patch("pantau.audio.tts.sd") as mock_sd,
        patch("pantau.audio.tts.sf") as mock_sf,
    ):
        mock_sf.read.return_value = (samples.astype(np.float32), sample_rate)
        tts._synthesize_and_play("Hallo")

    tts._voice.synthesize_wav.assert_called_once()
    mock_sd.play.assert_called_once()
    mock_sd.wait.assert_called_once()
