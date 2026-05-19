from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from pantau.audio.protocol import PartialResult, RecognitionResult
from pantau.config import SttConfig

logger = logging.getLogger(__name__)


def _ensure_simul_on_path() -> None:
    vendor = Path(__file__).parent.parent.parent.parent / "vendor" / "SimulStreaming"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


class SimulStreamingAdapter:
    """
    Streaming STT backend using SimulStreaming (SimulWhisper + AlignAtt).

    Records and transcribes concurrently: audio chunks are fed to
    SimulWhisperOnline while recording is still in progress, reducing
    end-to-end latency compared to batch-style adapters.

    Requires vendor/SimulStreaming (git submodule) and a Whisper .pt model.
    """

    def __init__(self, cfg: SttConfig) -> None:
        _ensure_simul_on_path()

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
