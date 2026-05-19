from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd

from pantau.audio.protocol import RecognitionResult
from pantau.config import SttConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = 512  # 32 ms — Silero VAD standard chunk size at 16 kHz
_MAX_DURATION_S = 10.0


def _to_frame_limit(seconds: float, chunk_frames: int) -> int:
    """Convert a duration in seconds to a frame count for silence detection."""
    return int(seconds * _SAMPLE_RATE / chunk_frames)


class BatchAdapter(ABC):
    """Base for batch STT adapters. Subclasses implement _transcribe()."""

    def __init__(self, cfg: SttConfig) -> None:
        from pantau.audio.vad import SileroVAD

        self._vad = SileroVAD()
        self._cfg = cfg

    @abstractmethod
    def _transcribe(self, audio: np.ndarray) -> str: ...

    def _record(self, initial_silence_timeout_s: float) -> np.ndarray:
        frames: list[np.ndarray] = []
        silence_frames = 0
        speech_started = False
        max_frames = _to_frame_limit(_MAX_DURATION_S, _CHUNK_FRAMES)
        initial_silence_limit = _to_frame_limit(
            initial_silence_timeout_s, _CHUNK_FRAMES
        )
        post_speech_silence_limit = _to_frame_limit(
            self._cfg.silence_stop_s, _CHUNK_FRAMES
        )

        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="float32"
        ) as stream:
            for _ in range(max_frames):
                chunk, _ = stream.read(_CHUNK_FRAMES)
                mono = chunk[:, 0]
                frames.append(mono.copy())
                if self._vad.is_speech(mono):
                    speech_started = True
                    silence_frames = 0
                else:
                    silence_frames += 1
                    limit = (
                        post_speech_silence_limit
                        if speech_started
                        else initial_silence_limit
                    )
                    if silence_frames >= limit:
                        break

        return np.concatenate(frames)

    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult:
        t0 = time.monotonic()
        audio = await asyncio.to_thread(self._record, initial_silence_timeout_s)
        text = await asyncio.to_thread(self._transcribe, audio)
        logger.debug("STT took %.2fs", time.monotonic() - t0)
        logger.info("STT transcribed: %s", text)
        return RecognitionResult(text=text)
