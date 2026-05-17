from __future__ import annotations

import platform
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pantau.config import SttConfig


@pytest.fixture
def cfg() -> SttConfig:
    return SttConfig(model_size="small", language="de", silence_stop_s=0.4)


def test_mlx_adapter_raises_on_non_apple_silicon() -> None:
    from pantau.audio.backends.mlx_whisper import MlxWhisperAdapter

    cfg = SttConfig()
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
        pytest.raises(RuntimeError, match="Apple Silicon"),
    ):
        MlxWhisperAdapter(cfg)


@pytest.mark.skipif(
    platform.system() != "Darwin" or "arm" not in platform.machine().lower(),
    reason="mlx_whisper requires Apple Silicon macOS",
)
def test_mlx_adapter_transcribe_calls_mlx_whisper(cfg: SttConfig) -> None:
    from pantau.audio.backends.mlx_whisper import MlxWhisperAdapter

    with patch.object(MlxWhisperAdapter, "__init__", return_value=None):
        adapter = MlxWhisperAdapter.__new__(MlxWhisperAdapter)
        adapter._cfg = cfg
        adapter._repo = "mlx-community/whisper-small-mlx"
        adapter._vad = MagicMock()

    mock_result = {"text": " Hallo Welt "}
    with patch("mlx_whisper.transcribe", return_value=mock_result) as mock_transcribe:
        result = adapter._transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hallo Welt"
    mock_transcribe.assert_called_once()


def test_mlx_model_mapping_small() -> None:
    from pantau.audio.backends.mlx_whisper import _MLX_MODELS

    assert "small" in _MLX_MODELS
    assert "mlx-community" in _MLX_MODELS["small"]


def test_mlx_model_mapping_fallback_to_small() -> None:
    from pantau.audio.backends.mlx_whisper import _MLX_MODELS

    assert _MLX_MODELS.get("unknown_size", _MLX_MODELS["small"]).endswith(
        "whisper-small-mlx"
    )
