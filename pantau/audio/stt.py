from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

from pantau.config import SttConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = 512  # 32 ms — silero VAD standard chunk size at 16 kHz
_SILENCE_STOP_S = 1.2
_MAX_DURATION_S = 10.0


class GermanSTT:
    def __init__(self, cfg: SttConfig) -> None:
        from faster_whisper import WhisperModel

        from pantau.audio.vad import SileroVAD

        device = cfg.device if cfg.device != "auto" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            cfg.model_size, device=device, compute_type=compute_type
        )
        self._vad = SileroVAD()
        self._language = cfg.language
        logger.debug(
            "GermanSTT model loaded: size=%s device=%s", cfg.model_size, device
        )

    def _record(self, initial_silence_timeout_s: float = _SILENCE_STOP_S) -> np.ndarray:
        frames: list[np.ndarray] = []
        silence_frames = 0
        speech_started = False
        max_frames = int(_MAX_DURATION_S * _SAMPLE_RATE / _CHUNK_FRAMES)
        initial_silence_limit = int(
            initial_silence_timeout_s * _SAMPLE_RATE / _CHUNK_FRAMES
        )
        post_speech_silence_limit = int(_SILENCE_STOP_S * _SAMPLE_RATE / _CHUNK_FRAMES)

        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="float32"
        ) as stream:
            for _ in range(max_frames):
                chunk, _ = stream.read(_CHUNK_FRAMES)
                mono = chunk[:, 0]
                frames.append(mono.copy())
                if self._vad.is_speech(mono):
                    speech_started = True
                    silence_frames = 0
                else:
                    silence_frames += 1
                    limit = (
                        post_speech_silence_limit
                        if speech_started
                        else initial_silence_limit
                    )
                    if silence_frames >= limit:
                        break

        return np.concatenate(frames)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = _SILENCE_STOP_S
    ) -> str:
        audio = await asyncio.to_thread(self._record, initial_silence_timeout_s)
        text = await asyncio.to_thread(self._transcribe, audio)
        logger.info("STT transcribed: %s", text)
        return text
