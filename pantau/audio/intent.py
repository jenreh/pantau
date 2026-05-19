from __future__ import annotations

import string
from typing import TYPE_CHECKING

from pantau.audio.protocol import RecognitionResult

if TYPE_CHECKING:
    from pantau.audio.protocol import STTAdapter

INTENT_RULES: dict[str, list[str]] = {
    "stop_session": ["beende dich", "auf wiedersehen", "exit"],
}


class RuleBasedIntentRecognizer:
    def classify(self, text: str) -> RecognitionResult:
        normalized = text.strip().lower().rstrip(string.punctuation)
        for intent, phrases in INTENT_RULES.items():
            if normalized in phrases:
                return RecognitionResult(text=text, intent=intent)
        return RecognitionResult(text=text)


class IntentAwareAdapter:
    def __init__(self, stt: STTAdapter, recognizer: RuleBasedIntentRecognizer) -> None:
        self._stt = stt
        self._recognizer = recognizer

    async def record_and_transcribe(
        self, initial_silence_timeout_s: float = 1.2
    ) -> RecognitionResult:
        result = await self._stt.record_and_transcribe(initial_silence_timeout_s)
        if result.text:
            return self._recognizer.classify(result.text)
        return result
