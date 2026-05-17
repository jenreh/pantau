from __future__ import annotations

from pantau.audio.backends.faster_whisper import FasterWhisperAdapter
from pantau.audio.intent import IntentAwareAdapter, RuleBasedIntentRecognizer
from pantau.audio.protocol import RecognitionResult, STTAdapter
from pantau.config import SttConfig

GermanSTT = FasterWhisperAdapter  # backward-compat alias for existing tests


def create_stt(cfg: SttConfig) -> STTAdapter:
    backend = _backend(cfg)
    if cfg.intent_enabled:
        return IntentAwareAdapter(backend, RuleBasedIntentRecognizer())
    return backend


def _backend(cfg: SttConfig) -> STTAdapter:
    match cfg.provider:
        case "vosk":
            from pantau.audio.backends.vosk import VoskAdapter

            return VoskAdapter(cfg)
        case "picovoice":
            from pantau.audio.backends.picovoice import PicovoiceCheetahAdapter

            return PicovoiceCheetahAdapter(cfg)
        case "mlx_whisper":
            from pantau.audio.backends.mlx_whisper import MlxWhisperAdapter

            return MlxWhisperAdapter(cfg)
        case _:
            return FasterWhisperAdapter(cfg)


__all__ = ["GermanSTT", "RecognitionResult", "STTAdapter", "create_stt"]
