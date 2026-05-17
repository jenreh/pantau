from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RecognitionResult:
    text: str
    intent: str | None = None


@runtime_checkable
class STTAdapter(Protocol):
    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult: ...
