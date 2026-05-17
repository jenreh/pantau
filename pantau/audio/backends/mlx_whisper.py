from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import platform
import time

import numpy as np

from pantau.audio.backends._batch_base import BatchAdapter
from pantau.audio.protocol import RecognitionResult
from pantau.config import SttConfig

log = logging.getLogger(__name__)

_MLX_MODELS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


class MlxWhisperAdapter(BatchAdapter):
    def __init__(self, cfg: SttConfig) -> None:
        if platform.system() != "Darwin" or "arm" not in platform.machine().lower():
            raise RuntimeError(
                "MlxWhisperAdapter requires Apple Silicon macOS (arm64 Darwin)"
            )
        super().__init__(cfg)
        self._repo = _MLX_MODELS.get(cfg.model_size, _MLX_MODELS["small"])
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-whisper"
        )
        log.info("Loading MLX Whisper model: %s", self._repo)
        self._executor.submit(self._load_model).result()
        log.info("MLX Whisper model ready")

    def _load_model(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        self._model = ModelHolder.get_model(self._repo, mx.float16)

    def _transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._repo,
            language=self._cfg.language,
            initial_prompt=self._cfg.initial_prompt or None,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.0,
            word_timestamps=False,
        )
        return result["text"].strip()

    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult:
        t0 = time.monotonic()
        audio = await asyncio.to_thread(self._record, initial_silence_timeout_s)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(self._executor, self._transcribe, audio)
        log.debug("STT took %.2fs", time.monotonic() - t0)
        log.info("STT transcribed: %s", text)
        return RecognitionResult(text=text)
