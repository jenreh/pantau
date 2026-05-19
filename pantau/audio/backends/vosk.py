from __future__ import annotations

import asyncio
import json
import logging
import time

import sounddevice as sd

from pantau.audio.backends._batch_base import _to_frame_limit
from pantau.audio.protocol import RecognitionResult
from pantau.config import SttConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = 480  # 30 ms at 16 kHz — int16 PCM for Vosk
_MAX_DURATION_S = 10.0


class VoskAdapter:
    def __init__(self, cfg: SttConfig) -> None:
        from vosk import KaldiRecognizer, Model

        model = Model(
            model_path=cfg.model_path if cfg.model_path else None,
            lang=cfg.language,
        )
        self._rec = KaldiRecognizer(model, _SAMPLE_RATE)
        self._cfg = cfg
        logger.debug(
            "VoskAdapter loaded: lang=%s model_path=%s",
            cfg.language,
            cfg.model_path,
        )

    def _run(self, initial_silence_timeout_s: float) -> RecognitionResult:
        speech_started = False
        silence_frames = 0
        initial_limit = _to_frame_limit(initial_silence_timeout_s, _CHUNK_FRAMES)
        post_limit = _to_frame_limit(self._cfg.silence_stop_s, _CHUNK_FRAMES)

        self._rec.Reset()
        t0 = time.monotonic()

        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="int16"
        ) as stream:
            while True:
                chunk, _ = stream.read(_CHUNK_FRAMES)
                mono_bytes = chunk[:, 0].tobytes()
                if self._rec.AcceptWaveform(mono_bytes):
                    r = json.loads(self._rec.Result())
                    text = r.get("text", "").strip()
                    logger.debug(
                        "STT (vosk) took %.2fs: %s", time.monotonic() - t0, text
                    )
                    logger.info("STT transcribed: %s", text)
                    return RecognitionResult(text=text)
                partial = json.loads(self._rec.PartialResult()).get("partial", "")
                if partial:
                    speech_started = True
                    silence_frames = 0
                else:
                    silence_frames += 1
                    limit = post_limit if speech_started else initial_limit
                    if silence_frames >= limit:
                        break

        r = json.loads(self._rec.FinalResult())
        text = r.get("text", "").strip()
        logger.debug("STT (vosk) took %.2fs: %s", time.monotonic() - t0, text)
        logger.info("STT transcribed: %s", text)
        return RecognitionResult(text=text)

    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult:
        return await asyncio.to_thread(self._run, initial_silence_timeout_s)
