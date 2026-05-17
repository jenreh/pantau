from __future__ import annotations

import numpy as np

from pantau.audio.backends._batch_base import BatchAdapter
from pantau.config import SttConfig


class FasterWhisperAdapter(BatchAdapter):
    def __init__(self, cfg: SttConfig) -> None:
        from faster_whisper import WhisperModel

        super().__init__(cfg)
        device = cfg.device if cfg.device != "auto" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            cfg.model_size, device=device, compute_type=compute_type
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=self._cfg.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
