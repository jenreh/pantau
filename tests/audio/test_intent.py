from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pantau.audio.intent import (
    INTENT_RULES,
    IntentAwareAdapter,
    RuleBasedIntentRecognizer,
)
from pantau.audio.protocol import RecognitionResult


@pytest.fixture
def recognizer() -> RuleBasedIntentRecognizer:
    return RuleBasedIntentRecognizer()


def test_classify_stop_session_phrase(recognizer: RuleBasedIntentRecognizer) -> None:
    result = recognizer.classify("Beende dich")
    assert result.intent == "stop_session"
    assert result.text == "Beende dich"


def test_classify_stop_session_with_punctuation(
    recognizer: RuleBasedIntentRecognizer,
) -> None:
    result = recognizer.classify("auf wiedersehen.")
    assert result.intent == "stop_session"


def test_classify_unknown_returns_no_intent(
    recognizer: RuleBasedIntentRecognizer,
) -> None:
    result = recognizer.classify("Schalte das Licht ein")
    assert result.intent is None
    assert result.text == "Schalte das Licht ein"


def test_classify_empty_string(recognizer: RuleBasedIntentRecognizer) -> None:
    result = recognizer.classify("")
    assert result.intent is None
    assert result.text == ""


def test_intent_rules_contains_stop_session() -> None:
    assert "stop_session" in INTENT_RULES
    assert len(INTENT_RULES["stop_session"]) > 0


@pytest.mark.asyncio
async def test_intent_aware_adapter_sets_intent() -> None:
    mock_stt = AsyncMock()
    mock_stt.record_and_transcribe.return_value = RecognitionResult(
        text="auf wiedersehen"
    )

    adapter = IntentAwareAdapter(mock_stt, RuleBasedIntentRecognizer())
    result = await adapter.record_and_transcribe()

    assert result.intent == "stop_session"
    assert result.text == "auf wiedersehen"


@pytest.mark.asyncio
async def test_intent_aware_adapter_passthrough_for_unknown() -> None:
    mock_stt = AsyncMock()
    mock_stt.record_and_transcribe.return_value = RecognitionResult(
        text="Mach das Licht an"
    )

    adapter = IntentAwareAdapter(mock_stt, RuleBasedIntentRecognizer())
    result = await adapter.record_and_transcribe()

    assert result.intent is None
    assert result.text == "Mach das Licht an"


@pytest.mark.asyncio
async def test_intent_aware_adapter_passthrough_for_empty() -> None:
    mock_stt = AsyncMock()
    mock_stt.record_and_transcribe.return_value = RecognitionResult(text="")

    adapter = IntentAwareAdapter(mock_stt, RuleBasedIntentRecognizer())
    result = await adapter.record_and_transcribe()

    assert result.text == ""
    assert result.intent is None
