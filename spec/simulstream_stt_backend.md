# Plan: SimulStreaming STT Backend (True Streaming)

## Context

Add `"simul_streaming"` as a new STT backend that **processes audio while recording** — the core latency win of SimulStreaming. Unlike the batch adapters (FasterWhisper, MlxWhisper), SimulStreaming's `SimulWhisperOnline` runs the AlignAtt decoder concurrently with audio capture, emitting partial results as audio arrives. This enables lower end-to-end latency and optional real-time UI feedback.

The existing `STTAdapter` protocol (`record_and_transcribe() → RecognitionResult`) is batch-oriented. True streaming requires either a new protocol or an adapter that bridges concurrent recording + processing internally and exposes partial results.

---

## Architecture Decision

**Introduce a `StreamingSTTAdapter` protocol** with an async generator interface, plus a `StreamingPipeline` helper that wires recording and SimulWhisper processing concurrently via `asyncio.Queue`.

`SimulStreamingAdapter` implements **both** `STTAdapter` and `StreamingSTTAdapter`:

- `record_and_transcribe()` — backward-compatible, wraps `stream_transcribe()`, returns final result
- `stream_transcribe()` — async generator, yields `PartialResult` items as they arrive

This keeps all existing callers unchanged while unlocking streaming for new callers (e.g., a live-feedback UI component or a future streaming pipeline stage).

---

## New Protocol Diagram

```
   sounddevice (16 kHz, float32)
         │  512-frame chunks
         ▼
   RecordingTask (thread)
     SileroVAD → speech/silence detection
         │  np.ndarray chunks
         ▼
   asyncio.Queue[np.ndarray | None]  ← None = end-of-speech sentinel
         │
         ▼
   ProcessingTask (thread)
     SimulWhisperOnline.insert_audio_chunk()
     SimulWhisperOnline.process_iter()  → PartialResult(text, is_final=False)
     SimulWhisperOnline.finish()        → PartialResult(text, is_final=True)
         │
         ▼
   asyncio.Queue[PartialResult | None]
         │
         ▼
   stream_transcribe() async generator  →  yields PartialResult
   record_and_transcribe()               →  collects, returns final RecognitionResult
```

---

## Critical Files

| File | Action |
|------|--------|
| `pantau/audio/protocol.py` | Add `PartialResult` dataclass + `StreamingSTTAdapter` protocol |
| `pantau/config.py` | Add `simul_*` fields to `SttConfig` |
| `pantau/audio/stt.py` | Add `"simul_streaming"` case to factory |
| `pantau/audio/backends/simul_streaming.py` | **Create** — `SimulStreamingAdapter` |
| `pantau/audio/backends/_streaming_pipeline.py` | **Create** — `StreamingPipeline` (queue-based recording ↔ processing bridge) |
| `tests/audio/backends/test_simul_streaming.py` | **Create** — unit tests |
| `pyproject.toml` | Add optional dependency group |

---

## Step-by-Step Implementation

### 0. Pre-check: Installability

```bash
curl -s https://raw.githubusercontent.com/ufal/SimulStreaming/main/pyproject.toml
curl -s https://raw.githubusercontent.com/ufal/SimulStreaming/main/setup.py
```

- **Has pyproject.toml / setup.py**: `uv add --optional simul_streaming "simulstreaming @ git+https://github.com/ufal/SimulStreaming.git"`
- **No installable package**: `git submodule add https://github.com/ufal/SimulStreaming vendor/SimulStreaming`, add `vendor/SimulStreaming` to `pythonpath` in `pyproject.toml`.

### 1. Protocol additions (`pantau/audio/protocol.py`)

```python
from dataclasses import dataclass, field


@dataclass
class PartialResult:
    text: str
    is_final: bool = False


@runtime_checkable
class StreamingSTTAdapter(Protocol):
    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult: ...

    def stream_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> AsyncIterator[PartialResult]: ...
```

`StreamingSTTAdapter` is a superset of `STTAdapter` — any `StreamingSTTAdapter` is also a valid `STTAdapter`.

### 2. SttConfig additions (`pantau/config.py`)

```python
simul_model_path: str = ""  # path to whisper .pt file (required)
simul_cif_ckpt_path: str = (
    ""  # CIF checkpoint (optional, improves word boundary detection)
)
simul_frame_threshold: int = 25  # AlignAtt attention threshold (frames × 0.02s each)
simul_audio_max_len: float = 30.0  # max audio buffer in seconds
simul_beams: int = 1  # beam width (1=greedy)
```

### 3. StreamingPipeline helper (`pantau/audio/backends/_streaming_pipeline.py`)

Encapsulates the concurrent recording ↔ processing coordination:

```python
import asyncio, threading
import numpy as np
import sounddevice as sd
from pantau.audio.vad import SileroVAD
from pantau.audio.protocol import PartialResult

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = 512


class StreamingPipeline:
    """
    Runs recording and SimulWhisper processing concurrently.
    Recording thread puts audio chunks into audio_q.
    Processing thread reads chunks, calls SimulWhisper, puts PartialResults into result_q.
    """

    def __init__(self, online, cfg) -> None:
        self._online = online
        self._cfg = cfg

    async def run(
        self, initial_silence_timeout_s: float
    ) -> AsyncIterator[PartialResult]:
        audio_q: asyncio.Queue = asyncio.Queue()
        result_q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        rec_thread = threading.Thread(
            target=self._record_worker,
            args=(audio_q, loop, initial_silence_timeout_s),
            daemon=True,
        )
        proc_thread = threading.Thread(
            target=self._process_worker,
            args=(audio_q, result_q, loop),
            daemon=True,
        )
        rec_thread.start()
        proc_thread.start()

        while True:
            item = await result_q.get()
            if item is None:
                break
            yield item

    def _record_worker(self, audio_q, loop, initial_silence_timeout_s):
        """Records audio with SileroVAD; puts chunks then None sentinel into audio_q."""
        vad = SileroVAD()
        silence_frames = speech_started = 0
        initial_limit = int(initial_silence_timeout_s * _SAMPLE_RATE / _CHUNK_FRAMES)
        post_limit = int(self._cfg.silence_stop_s * _SAMPLE_RATE / _CHUNK_FRAMES)

        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="float32"
        ) as stream:
            while True:
                chunk, _ = stream.read(_CHUNK_FRAMES)
                mono = chunk[:, 0].copy()
                asyncio.run_coroutine_threadsafe(audio_q.put(mono), loop)
                if vad.is_speech(mono):
                    speech_started = True
                    silence_frames = 0
                else:
                    silence_frames += 1
                    limit = post_limit if speech_started else initial_limit
                    if silence_frames >= limit:
                        break

        asyncio.run_coroutine_threadsafe(audio_q.put(None), loop)

    def _process_worker(self, audio_q, result_q, loop):
        """Reads audio chunks, feeds to SimulWhisper, puts PartialResults into result_q."""
        self._online.init()

        async def get():
            return await audio_q.get()

        while True:
            future = asyncio.run_coroutine_threadsafe(get(), loop)
            chunk = future.result()
            if chunk is None:
                result = self._online.finish()
                text = result.get("text", "").strip() if result else ""
                asyncio.run_coroutine_threadsafe(
                    result_q.put(PartialResult(text=text, is_final=True)), loop
                )
                break
            self._online.insert_audio_chunk(chunk)
            partial = self._online.process_iter()
            if partial and partial.get("text", "").strip():
                asyncio.run_coroutine_threadsafe(
                    result_q.put(PartialResult(text=partial["text"].strip())), loop
                )

        asyncio.run_coroutine_threadsafe(result_q.put(None), loop)
```

### 4. SimulStreamingAdapter (`pantau/audio/backends/simul_streaming.py`)

```python
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from pantau.audio.protocol import PartialResult, RecognitionResult
from pantau.config import SttConfig

logger = logging.getLogger(__name__)


class SimulStreamingAdapter:
    def __init__(self, cfg: SttConfig) -> None:
        from simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline

        asr = SimulWhisperASR(
            language=cfg.language,
            model_path=cfg.simul_model_path,
            cif_ckpt_path=cfg.simul_cif_ckpt_path or None,
            frame_threshold=cfg.simul_frame_threshold,
            audio_max_len=cfg.simul_audio_max_len,
            audio_min_len=0.0,
            segment_length=1.0,
            beams=cfg.simul_beams,
            task="transcribe",
            decoder_type="greedy",
            never_fire=False,
            init_prompt=cfg.initial_prompt or None,
            static_init_prompt=None,
            max_context_tokens=None,
            logdir=None,
        )
        self._online = SimulWhisperOnline(asr)
        self._cfg = cfg
        logger.debug(
            "SimulStreamingAdapter loaded: lang=%s model=%s",
            cfg.language,
            cfg.simul_model_path,
        )

    async def stream_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> AsyncIterator[PartialResult]:
        from pantau.audio.backends._streaming_pipeline import StreamingPipeline

        pipeline = StreamingPipeline(self._online, self._cfg)
        async for result in pipeline.run(initial_silence_timeout_s):
            yield result

    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult:
        text = ""
        async for partial in self.stream_transcribe(initial_silence_timeout_s):
            if partial.is_final:
                text = partial.text
        logger.info("STT transcribed: %s", text)
        return RecognitionResult(text=text)
```

### 5. Factory registration (`pantau/audio/stt.py`)

```python
case "simul_streaming":
    from pantau.audio.backends.simul_streaming import SimulStreamingAdapter
    return SimulStreamingAdapter(cfg)
```

### 6. Tests (`tests/audio/backends/test_simul_streaming.py`)

Patterns from `test_vosk.py`:

- Bypass `__init__` with `patch.object(SimulStreamingAdapter, "__init__", return_value=None)`
- Mock `_online` with `MagicMock`
- `test_record_and_transcribe_returns_final_text`
- `test_stream_transcribe_yields_partial_then_final`
- `test_online_init_called_before_each_run` — `_online.init()` asserted
- `test_empty_finish_returns_empty_string` — `finish()` returns `{}` → `""`
- `test_factory_creates_simul_streaming_adapter`

For `_streaming_pipeline.py`, integration-level tests mocking `sd.InputStream` and `SileroVAD`.

---

## Wake Word Trigger Compatibility

Wake word detection calls `record_and_transcribe()` — the same method all existing adapters implement. `SimulStreamingAdapter.record_and_transcribe()` is fully implemented (it wraps `stream_transcribe()` and returns the final `RecognitionResult`). The wake-word → STT → intent pipeline is **unchanged**.

`stream_transcribe()` is additive: only callers that explicitly want partial results (e.g. a future live-feedback UI component) use it.

---

## Key Notes

- **GPU strongly recommended**: SimulStreaming with PyTorch Whisper large-v3 needs ~10 GB VRAM for real-time. CPU works but too slow. No guard needed — let PyTorch raise its own errors.
- **Model file**: `simul_model_path` → Whisper `.pt` file. No auto-download; user must provide path.
- **CIF checkpoint**: optional. Without it, AlignAtt uses pure attention-based word detection.
- **Thread safety**: `_streaming_pipeline.py` bridges sync threads ↔ async event loop via `asyncio.run_coroutine_threadsafe`. This is the pattern used by the existing adapters (`asyncio.to_thread`).
- **`process_iter()` frequency**: called after every 512-frame chunk (32 ms). SimulStreaming may return `{}` for most calls and only emit on attention boundaries — this is expected behavior.

---

## Verification

```bash
task sync
task test
task lint && task format

# Manual streaming test (needs model file):
python -c "
import asyncio
from pantau.config import SttConfig
from pantau.audio.backends.simul_streaming import SimulStreamingAdapter

cfg = SttConfig(provider='simul_streaming', simul_model_path='large-v3.pt')
adapter = SimulStreamingAdapter(cfg)

async def main():
    async for partial in adapter.stream_transcribe():
        print('partial' if not partial.is_final else 'FINAL', repr(partial.text))

asyncio.run(main())
"
```
