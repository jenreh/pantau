from __future__ import annotations

import platform

import numpy as np

from pantau.audio.backends._batch_base import BatchAdapter
from pantau.config import SttConfig

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

    def _transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._repo,
            language=self._cfg.language,
            verbose=False,
        )
        return result["text"].strip()
