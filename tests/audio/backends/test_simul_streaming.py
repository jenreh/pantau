from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

from pantau.audio.protocol import PartialResult, RecognitionResult
from pantau.config import SttConfig


@pytest.fixture
def cfg() -> SttConfig:
    return SttConfig(
        provider="simul_streaming",
        language="de",
        silence_stop_s=0.4,
        simul_model_path="fake/large-v3.pt",
    )


@pytest.fixture
def adapter(cfg: SttConfig) -> object:
    from pantau.audio.backends.simul_streaming import SimulStreamingAdapter

    with patch.object(SimulStreamingAdapter, "__init__", return_value=None):
        instance = SimulStreamingAdapter.__new__(SimulStreamingAdapter)
        instance._cfg = cfg
        instance._online = MagicMock()
        instance._online.init = MagicMock()
        instance._online.insert_audio_chunk = MagicMock()
        instance._online.process_iter = MagicMock(return_value={})
        instance._online.finish = MagicMock(return_value={"text": "Licht einschalten"})
        yield instance


async def _collect(gen) -> list[PartialResult]:
    results = []
    async for item in gen:
        results.append(item)
    return results


def test_record_and_transcribe_returns_final_text(adapter: object) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield PartialResult(text="Licht", is_final=False)
        yield PartialResult(text="Licht einschalten", is_final=True)

    with patch.object(adapter, "stream_transcribe", side_effect=fake_stream):
        result = asyncio.run(adapter.record_and_transcribe())

    assert isinstance(result, RecognitionResult)
    assert result.text == "Licht einschalten"


def test_record_and_transcribe_ignores_partial_text(adapter: object) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield PartialResult(text="wrong partial", is_final=False)
        yield PartialResult(text="correct final", is_final=True)

    with patch.object(adapter, "stream_transcribe", side_effect=fake_stream):
        result = asyncio.run(adapter.record_and_transcribe())

    assert result.text == "correct final"


def test_record_and_transcribe_empty_result(adapter: object) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield PartialResult(text="", is_final=True)

    with patch.object(adapter, "stream_transcribe", side_effect=fake_stream):
        result = asyncio.run(adapter.record_and_transcribe())

    assert result.text == ""


def test_factory_creates_simul_streaming_adapter(cfg: SttConfig) -> None:
    from pantau.audio.backends.simul_streaming import SimulStreamingAdapter
    from pantau.audio.stt import create_stt

    with patch.object(SimulStreamingAdapter, "__init__", return_value=None):
        adapter = create_stt(cfg)

    assert isinstance(adapter, SimulStreamingAdapter)


def test_ensure_simul_on_path_adds_vendor_if_present(
    tmp_path: pytest.TempPathFactory,
) -> None:
    from pantau.audio.backends.simul_streaming import _ensure_simul_on_path

    fake_vendor = tmp_path / "vendor" / "SimulStreaming"
    fake_vendor.mkdir(parents=True)

    with patch(
        "pantau.audio.backends.simul_streaming.Path.__truediv__",
        return_value=fake_vendor,
    ):
        original_path = sys.path.copy()
        _ensure_simul_on_path()
        assert str(fake_vendor) in sys.path
        sys.path[:] = original_path


def test_ensure_simul_on_path_skips_if_already_present(
    tmp_path: pytest.TempPathFactory,
) -> None:
    from pantau.audio.backends.simul_streaming import _ensure_simul_on_path

    fake_vendor = tmp_path / "vendor" / "SimulStreaming"
    fake_vendor.mkdir(parents=True)
    sys.path.insert(0, str(fake_vendor))

    with patch(
        "pantau.audio.backends.simul_streaming.Path.__truediv__",
        return_value=fake_vendor,
    ):
        count_before = sys.path.count(str(fake_vendor))
        _ensure_simul_on_path()
        assert sys.path.count(str(fake_vendor)) == count_before

    sys.path.remove(str(fake_vendor))


class TestStreamingPipeline:
    @pytest.fixture
    def online(self) -> MagicMock:
        mock = MagicMock()
        mock.init = MagicMock()
        mock.insert_audio_chunk = MagicMock()
        mock.process_iter = MagicMock(return_value={})
        mock.finish = MagicMock(return_value={"text": "Hallo Welt"})
        return mock

    async def _run_worker(self, pipeline, audio_q, result_q) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(pipeline._process_worker, audio_q, result_q, loop)

    async def test_process_worker_calls_online_init(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        from pantau.audio.backends._streaming_pipeline import StreamingPipeline

        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        audio_q.put(None)

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        online.init.assert_called_once()

    async def test_process_worker_calls_finish_on_sentinel(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        from pantau.audio.backends._streaming_pipeline import StreamingPipeline

        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        audio_q.put(None)

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        online.finish.assert_called_once()

    async def test_process_worker_emits_final_result(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        from pantau.audio.backends._streaming_pipeline import StreamingPipeline

        online.finish.return_value = {"text": "  Küche einschalten  "}
        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        audio_q.put(None)

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        items = []
        while not result_q.empty():
            items.append(result_q.get_nowait())

        final = next(i for i in items if isinstance(i, PartialResult) and i.is_final)
        assert final.text == "Küche einschalten"

    async def test_process_worker_emits_partial_results(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        import numpy as np

        from pantau.audio.backends._streaming_pipeline import StreamingPipeline

        online.process_iter.side_effect = [
            {"text": "Kü"},
            {"text": "Küche"},
            {},
        ]
        online.finish.return_value = {"text": "Küche einschalten"}

        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        chunk = np.zeros(512, dtype=np.float32)
        audio_q.put(chunk)
        audio_q.put(chunk)
        audio_q.put(chunk)
        audio_q.put(None)

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        items = []
        while not result_q.empty():
            items.append(result_q.get_nowait())

        partials = [i for i in items if isinstance(i, PartialResult) and not i.is_final]
        assert len(partials) == 2
        assert partials[0].text == "Kü"
        assert partials[1].text == "Küche"
