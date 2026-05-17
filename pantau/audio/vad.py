from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SileroVAD:
    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._model.eval()
        logger.debug("SileroVAD model loaded")

    def is_speech(self, chunk: np.ndarray) -> bool:
        import torch

        tensor = torch.from_numpy(chunk).float()
        confidence: float = self._model(tensor, 16000).item()
        return confidence >= 0.5
