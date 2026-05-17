from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def vad() -> object:
    from pantau.audio.vad import SileroVAD

    instance = SileroVAD.__new__(SileroVAD)
    instance._model = MagicMock()
    return instance


def test_is_speech_returns_true_when_confidence_high(vad: object) -> None:
    vad._model.return_value = MagicMock(**{"item.return_value": 0.9})

    result = vad.is_speech(np.zeros(480, dtype=np.float32))

    assert result is True


def test_is_speech_returns_false_when_confidence_low(vad: object) -> None:
    vad._model.return_value = MagicMock(**{"item.return_value": 0.1})

    result = vad.is_speech(np.zeros(480, dtype=np.float32))

    assert result is False


def test_is_speech_threshold_boundary(vad: object) -> None:
    vad._model.return_value = MagicMock(**{"item.return_value": 0.5})

    result = vad.is_speech(np.zeros(480, dtype=np.float32))

    assert result is True
