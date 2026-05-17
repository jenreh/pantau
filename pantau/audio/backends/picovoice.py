from __future__ import annotations

import asyncio
import logging
import time

import sounddevice as sd

from pantau.audio.backends._batch_base import _to_frame_limit
from pantau.audio.protocol import RecognitionResult
from pantau.config import SttConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000


class PicovoiceCheetahAdapter:
    def __init__(self, cfg: SttConfig) -> None:
        import pvcheetah

        self._cheetah = pvcheetah.create(
            access_key=cfg.api_key,
            endpoint_duration_sec=cfg.silence_stop_s,
            enable_automatic_punctuation=False,
        )
        self._cfg = cfg
        logger.debug("PicovoiceCheetahAdapter loaded")

    def _run(self, initial_silence_timeout_s: float) -> RecognitionResult:
        frame_length = self._cheetah.frame_length
        initial_limit_frames = _to_frame_limit(initial_silence_timeout_s, frame_length)
        transcript: list[str] = []
        silent_frames = 0
        speech_started = False
        t0 = time.monotonic()

        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="int16"
        ) as stream:
            while True:
                chunk, _ = stream.read(frame_length)
                partial, is_endpoint = self._cheetah.process(chunk[:, 0])
                if partial:
                    transcript.append(partial)
                    speech_started = True
                    silent_frames = 0
                else:
                    silent_frames += 1
                if is_endpoint:
                    final = self._cheetah.flush()
                    if final:
                        transcript.append(final)
                    break
                if not speech_started and silent_frames >= initial_limit_frames:
                    break

        text = "".join(transcript).strip()
        logger.debug("STT (picovoice) took %.2fs: %s", time.monotonic() - t0, text)
        logger.info("STT transcribed: %s", text)
        return RecognitionResult(text=text)

    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult:
        return await asyncio.to_thread(self._run, initial_silence_timeout_s)
