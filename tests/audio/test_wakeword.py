from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from pantau.config import WakeWordConfig


@pytest.fixture
def ww_cfg() -> WakeWordConfig:
    return WakeWordConfig(model="models/pantau.tflite", threshold=0.5)


@pytest.fixture
def listener(ww_cfg: WakeWordConfig) -> object:
    from pantau.audio.wakeword import WakeWordListener

    with patch("pantau.audio.wakeword.WakeWordListener.__init__") as mock_init:
        mock_init.return_value = None
        instance = WakeWordListener.__new__(WakeWordListener)
        instance._model = MagicMock()
        instance._threshold = ww_cfg.threshold
        instance._stop_event = threading.Event()
        yield instance


@pytest.mark.asyncio
async def test_listen_returns_when_wake_word_detected(listener: object) -> None:
    call_count = 0

    def fake_detect() -> None:
        nonlocal call_count
        call_count += 1

    listener._detect = fake_detect

    with patch(
        "pantau.audio.wakeword.asyncio.to_thread", side_effect=lambda fn, *a: fn(*a)
    ):
        await listener.listen()

    assert call_count == 1


def test_detect_exits_on_threshold_met(listener: object) -> None:
    predictions = [
        {"model": 0.1},
        {"model": 0.3},
        {"model": 0.8},
    ]
    predict_iter = iter(predictions)

    chunk_mock = MagicMock()
    chunk_mock.__getitem__ = MagicMock(return_value=MagicMock())

    fake_stream = MagicMock()
    fake_stream.__enter__ = MagicMock(return_value=fake_stream)
    fake_stream.__exit__ = MagicMock(return_value=False)
    fake_stream.read.return_value = (
        MagicMock(__getitem__=MagicMock(return_value=MagicMock())),
        None,
    )

    listener._model.predict.side_effect = lambda _: next(predict_iter)

    with patch("pantau.audio.wakeword.sd") as mock_sd:
        mock_sd.InputStream.return_value = fake_stream
        reads = 0

        def fake_read(frames: int):
            import numpy as np

            nonlocal reads
            reads += 1
            arr = MagicMock()
            arr.__getitem__ = MagicMock(return_value=np.zeros(frames, dtype=np.int16))
            return arr, None

        fake_stream.read.side_effect = fake_read

        listener._detect()

    assert reads == 3
    listener._model.reset.assert_called_once()
