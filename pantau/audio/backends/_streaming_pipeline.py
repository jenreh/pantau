from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from pantau.audio.protocol import PartialResult

if TYPE_CHECKING:
    from pantau.config import SttConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = 512


def _ensure_simul_on_path() -> None:
    vendor = Path(__file__).parent.parent.parent.parent / "vendor" / "SimulStreaming"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


class StreamingPipeline:
    """
    Concurrent recording + SimulWhisper processing pipeline.

    Two daemon threads run in parallel:
    - Recording thread: captures 16 kHz audio via sounddevice, applies SileroVAD
      for silence detection, and enqueues numpy chunks into a thread-safe queue.
    - Processing thread: dequeues chunks, feeds them to SimulWhisperOnline, and
      posts PartialResult items to an asyncio queue consumed by `run()`.

    The None sentinel in audio_q signals end-of-speech; None in result_q signals
    that the async generator should stop.
    """

    def __init__(self, online: object, cfg: SttConfig) -> None:
        self._online = online
        self._cfg = cfg

    async def run(
        self, initial_silence_timeout_s: float
    ) -> AsyncGenerator[PartialResult]:
        loop = asyncio.get_running_loop()
        audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
        result_q: asyncio.Queue[PartialResult | None] = asyncio.Queue()

        rec = threading.Thread(
            target=self._record_worker,
            args=(audio_q, initial_silence_timeout_s),
            daemon=True,
        )
        proc = threading.Thread(
            target=self._process_worker,
            args=(audio_q, result_q, loop),
            daemon=True,
        )
        rec.start()
        proc.start()

        while True:
            item = await result_q.get()
            if item is None:
                break
            yield item

    def _record_worker(
        self,
        audio_q: queue.Queue[np.ndarray | None],
        initial_silence_timeout_s: float,
    ) -> None:
        from pantau.audio.vad import SileroVAD

        vad = SileroVAD()
        silence_frames = 0
        speech_started = False
        initial_limit = int(initial_silence_timeout_s * _SAMPLE_RATE / _CHUNK_FRAMES)
        post_limit = int(self._cfg.silence_stop_s * _SAMPLE_RATE / _CHUNK_FRAMES)

        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="float32"
        ) as stream:
            while True:
                chunk, _ = stream.read(_CHUNK_FRAMES)
                mono = chunk[:, 0].copy()
                audio_q.put(mono)
                if vad.is_speech(mono):
                    speech_started = True
                    silence_frames = 0
                else:
                    silence_frames += 1
                    limit = post_limit if speech_started else initial_limit
                    if silence_frames >= limit:
                        break

        audio_q.put(None)
        logger.debug("StreamingPipeline: recording finished")

    def _process_worker(
        self,
        audio_q: queue.Queue[np.ndarray | None],
        result_q: asyncio.Queue[PartialResult | None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        _ensure_simul_on_path()
        self._online.init()

        while True:
            chunk = audio_q.get()
            if chunk is None:
                result = self._online.finish()
                text = result.get("text", "").strip() if result else ""
                asyncio.run_coroutine_threadsafe(
                    result_q.put(PartialResult(text=text, is_final=True)), loop
                ).result()
                break

            self._online.insert_audio_chunk(chunk)

        asyncio.run_coroutine_threadsafe(result_q.put(None), loop).result()
        logger.debug("StreamingPipeline: processing finished")
