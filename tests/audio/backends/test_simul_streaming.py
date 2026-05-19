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


def test_init_creates_adapter_with_mocked_simul(cfg: SttConfig) -> None:
    from pantau.audio.backends.simul_streaming import SimulStreamingAdapter

    mock_asr_cls = MagicMock()
    mock_online_cls = MagicMock()
    mock_asr = mock_asr_cls.return_value
    mock_online = mock_online_cls.return_value

    mock_module = MagicMock()
    mock_module.SimulWhisperASR = mock_asr_cls
    mock_module.SimulWhisperOnline = mock_online_cls

    import sys

    with patch.dict(sys.modules, {"simulstreaming_whisper": mock_module}):
        adapter = SimulStreamingAdapter(cfg)

    assert adapter._online is mock_online
    assert adapter._cfg is cfg
    mock_asr_cls.assert_called_once()
    mock_online_cls.assert_called_once_with(mock_asr)


async def test_stream_transcribe_yields_from_pipeline(adapter: object) -> None:
    expected = [
        PartialResult(text="partial", is_final=False),
        PartialResult(text="final", is_final=True),
    ]

    async def fake_run(initial_silence_timeout_s: float) -> None:
        for item in expected:
            yield item

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run = fake_run
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)

    with patch(
        "pantau.audio.backends._streaming_pipeline.StreamingPipeline",
        mock_pipeline_cls,
    ):
        results = await _collect(adapter.stream_transcribe())

    assert results == expected


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

        from pantau.audio.backends._streaming_pipeline import (
            StreamingPipeline,
            _EndMarker,
        )

        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        audio_q.put(_EndMarker(speech_detected=True))

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        online.init.assert_called_once()

    async def test_process_worker_calls_finish_on_sentinel(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        from pantau.audio.backends._streaming_pipeline import (
            StreamingPipeline,
            _EndMarker,
        )

        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        audio_q.put(_EndMarker(speech_detected=True))

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        online.finish.assert_called_once()

    async def test_process_worker_emits_final_result(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        from pantau.audio.backends._streaming_pipeline import (
            StreamingPipeline,
            _EndMarker,
        )

        online.finish.return_value = {"text": "  Küche einschalten  "}
        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        audio_q.put(_EndMarker(speech_detected=True))

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        items = []
        while not result_q.empty():
            items.append(result_q.get_nowait())

        final = next(i for i in items if isinstance(i, PartialResult) and i.is_final)
        assert final.text == "Küche einschalten"

    async def test_run_yields_results_and_stops_on_sentinel(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue as _queue

        from pantau.audio.backends._streaming_pipeline import StreamingPipeline

        pipeline = StreamingPipeline(online, cfg)

        def fake_process_worker(
            audio_q: _queue.Queue,
            result_q: asyncio.Queue,
            loop: asyncio.AbstractEventLoop,
        ) -> None:
            asyncio.run_coroutine_threadsafe(
                result_q.put(PartialResult(text="partial", is_final=False)), loop
            ).result()
            asyncio.run_coroutine_threadsafe(
                result_q.put(PartialResult(text="final text", is_final=True)), loop
            ).result()
            asyncio.run_coroutine_threadsafe(result_q.put(None), loop).result()

        with (
            patch.object(pipeline, "_record_worker"),
            patch.object(pipeline, "_process_worker", side_effect=fake_process_worker),
        ):
            results = []
            async for item in pipeline.run(1.2):
                results.append(item)

        assert len(results) == 2
        assert results[0].text == "partial"
        assert results[0].is_final is False
        assert results[1].text == "final text"
        assert results[1].is_final is True

    def test_record_worker_stops_after_post_speech_silence(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue as _queue

        import numpy as np

        from pantau.audio.backends._streaming_pipeline import (
            _CHUNK_FRAMES,
            _SAMPLE_RATE,
            StreamingPipeline,
            _EndMarker,
        )

        pipeline = StreamingPipeline(online, cfg)
        audio_q: _queue.Queue = _queue.Queue()

        post_limit = int(cfg.silence_stop_s * _SAMPLE_RATE / _CHUNK_FRAMES)
        speech_chunk = np.ones((512, 1), dtype=np.float32)
        silence_chunk = np.zeros((512, 1), dtype=np.float32)
        read_returns = [(speech_chunk, None)] + [(silence_chunk, None)] * (
            post_limit + 1
        )

        mock_stream = MagicMock()
        mock_stream.read.side_effect = read_returns

        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = [True] + [False] * (post_limit + 1)

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "pantau.audio.backends._streaming_pipeline.sd.InputStream",
                return_value=mock_ctx,
            ),
            patch("pantau.audio.vad.SileroVAD", return_value=mock_vad),
        ):
            pipeline._record_worker(audio_q, 1.2)

        items = []
        while not audio_q.empty():
            items.append(audio_q.get_nowait())

        assert isinstance(items[-1], _EndMarker)
        assert items[-1].speech_detected is True
        assert len(items) >= 2

    async def test_process_worker_batches_chunks_no_partials(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        import numpy as np

        from pantau.audio.backends._streaming_pipeline import (
            StreamingPipeline,
            _EndMarker,
        )

        online.finish.return_value = {"text": "Küche einschalten"}

        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        chunk = np.zeros(512, dtype=np.float32)
        audio_q.put(chunk)
        audio_q.put(chunk)
        audio_q.put(chunk)
        audio_q.put(_EndMarker(speech_detected=True))

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        online.process_iter.assert_not_called()
        assert online.insert_audio_chunk.call_count == 3

        items = []
        while not result_q.empty():
            items.append(result_q.get_nowait())

        partials = [i for i in items if isinstance(i, PartialResult) and not i.is_final]
        assert len(partials) == 0

        finals = [i for i in items if isinstance(i, PartialResult) and i.is_final]
        assert len(finals) == 1
        assert finals[0].text == "Küche einschalten"

    async def test_process_worker_skips_finish_when_no_speech_detected(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue

        from pantau.audio.backends._streaming_pipeline import (
            StreamingPipeline,
            _EndMarker,
        )

        audio_q: queue.Queue = queue.Queue()
        result_q: asyncio.Queue = asyncio.Queue()

        audio_q.put(_EndMarker(speech_detected=False))

        pipeline = StreamingPipeline(online, cfg)
        await self._run_worker(pipeline, audio_q, result_q)

        online.finish.assert_not_called()

        items = []
        while not result_q.empty():
            items.append(result_q.get_nowait())

        finals = [i for i in items if isinstance(i, PartialResult) and i.is_final]
        assert len(finals) == 1
        assert finals[0].text == ""

    def test_record_worker_emits_marker_with_speech_detected_false_on_initial_timeout(
        self, online: MagicMock, cfg: SttConfig
    ) -> None:
        import queue as _queue

        import numpy as np

        from pantau.audio.backends._streaming_pipeline import (
            _CHUNK_FRAMES,
            _SAMPLE_RATE,
            StreamingPipeline,
            _EndMarker,
        )

        pipeline = StreamingPipeline(online, cfg)
        audio_q: _queue.Queue = _queue.Queue()

        initial_timeout_s = 3.0
        initial_limit = int(initial_timeout_s * _SAMPLE_RATE / _CHUNK_FRAMES)
        silence_chunk = np.zeros((512, 1), dtype=np.float32)
        read_returns = [(silence_chunk, None)] * (initial_limit + 1)

        mock_stream = MagicMock()
        mock_stream.read.side_effect = read_returns

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "pantau.audio.backends._streaming_pipeline.sd.InputStream",
                return_value=mock_ctx,
            ),
            patch("pantau.audio.vad.SileroVAD", return_value=mock_vad),
        ):
            pipeline._record_worker(audio_q, initial_timeout_s)

        items = []
        while not audio_q.empty():
            items.append(audio_q.get_nowait())

        assert isinstance(items[-1], _EndMarker)
        assert items[-1].speech_detected is False
