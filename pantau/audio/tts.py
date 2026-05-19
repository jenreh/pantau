from __future__ import annotations

import asyncio
import io
import logging
import wave

import sounddevice as sd
import soundfile as sf

from pantau.config import TtsConfig

logger = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, cfg: TtsConfig) -> None:
        from piper.config import SynthesisConfig
        from piper.voice import PiperVoice

        self._voice = PiperVoice.load(cfg.model)
        self._syn_config = SynthesisConfig(length_scale=1.0 / cfg.speak_rate)
        logger.debug(
            "PiperTTS model loaded: model=%s rate=%s", cfg.model, cfg.speak_rate
        )

    def _synthesize_and_play(self, text: str) -> None:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file, syn_config=self._syn_config)

        buf.seek(0)
        data, samplerate = sf.read(buf, dtype="float32")
        sd.play(data, samplerate=samplerate)
        sd.wait()

    async def speak(self, text: str) -> None:
        logger.debug("TTS speak: %s", text)
        await asyncio.to_thread(self._synthesize_and_play, text)
