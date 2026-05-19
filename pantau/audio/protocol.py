from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class RecognitionResult:
    text: str
    intent: str | None = None


@dataclass
class PartialResult:
    text: str
    is_final: bool = False


@runtime_checkable
class STTAdapter(Protocol):
    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult: ...


class StreamingSTTAdapter(STTAdapter, Protocol):
    def stream_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> AsyncIterator[PartialResult]: ...
