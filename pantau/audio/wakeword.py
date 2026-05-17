from __future__ import annotations

import asyncio
import logging
import threading

import sounddevice as sd

from pantau.config import WakeWordConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = int(_SAMPLE_RATE * 0.08)  # 80 ms chunks


class WakeWordListener:
    def __init__(self, cfg: WakeWordConfig) -> None:
        from pathlib import Path

        from openwakeword.model import Model

        model_paths: list[str] = []
        if cfg.model and Path(cfg.model).exists():
            model_paths = [cfg.model]
        elif cfg.model:
            logger.warning(
                "Wake word model not found at %s; falling back to pretrained models",
                cfg.model,
            )

        self._model = Model(wakeword_models=model_paths, inference_framework="onnx")
        self._threshold = cfg.threshold
        self._stop_event = threading.Event()
        logger.debug(
            "WakeWordListener loaded: models=%s threshold=%s",
            model_paths or "pretrained",
            cfg.threshold,
        )

    def _detect(self) -> None:
        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="int16"
        ) as stream:
            while not self._stop_event.is_set():
                chunk, _ = stream.read(_CHUNK_FRAMES)
                audio = chunk[:, 0]
                predictions = self._model.predict(audio)
                scores = predictions.values()
                if any(score >= self._threshold for score in scores):
                    logger.info("Wake word detected")
                    return

    def stop(self) -> None:
        self._stop_event.set()

    async def listen(self) -> None:
        await asyncio.to_thread(self._detect)
