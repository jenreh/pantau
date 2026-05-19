from __future__ import annotations

import asyncio
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
def test_mlx_adapter_preloads_model_in_dedicated_thread(cfg: SttConfig) -> None:
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    from pantau.audio.backends.mlx_whisper import MlxWhisperAdapter

    mock_model = MagicMock()
    with (
        patch.object(ModelHolder, "get_model", return_value=mock_model) as mock_get,
        patch("mlx_whisper.transcribe", return_value={"text": ""}),
    ):
        adapter = MlxWhisperAdapter(cfg)

    mock_get.assert_called_once_with("mlx-community/whisper-small-mlx", mx.float16)
    assert adapter._model is mock_model
    assert adapter._executor._max_workers == 1


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


@pytest.mark.skipif(
    platform.system() != "Darwin" or "arm" not in platform.machine().lower(),
    reason="mlx_whisper requires Apple Silicon macOS",
)
def test_mlx_record_and_transcribe_uses_dedicated_executor(cfg: SttConfig) -> None:
    import concurrent.futures

    from pantau.audio.backends.mlx_whisper import MlxWhisperAdapter

    with patch.object(MlxWhisperAdapter, "__init__", return_value=None):
        adapter = MlxWhisperAdapter.__new__(MlxWhisperAdapter)
        adapter._cfg = cfg
        adapter._repo = "mlx-community/whisper-small-mlx"
        adapter._vad = MagicMock()
        adapter._model = MagicMock()
        adapter._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    audio = np.zeros(16000, dtype=np.float32)

    with (
        patch.object(adapter, "_record", return_value=audio),
        patch.object(adapter, "_transcribe", return_value="Hallo") as mock_transcribe,
    ):
        result = asyncio.run(adapter.record_and_transcribe())

    assert result.text == "Hallo"
    mock_transcribe.assert_called_once_with(audio)


def test_mlx_model_mapping_small() -> None:
    from pantau.audio.backends.mlx_whisper import _MLX_MODELS

    assert "small" in _MLX_MODELS
    assert "mlx-community" in _MLX_MODELS["small"]


def test_mlx_model_mapping_fallback_to_small() -> None:
    from pantau.audio.backends.mlx_whisper import _MLX_MODELS

    assert _MLX_MODELS.get("unknown_size", _MLX_MODELS["small"]).endswith(
        "whisper-small-mlx"
    )
