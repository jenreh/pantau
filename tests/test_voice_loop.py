from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pantau.audio.protocol import RecognitionResult


def _make_session_mock(process_side_effects: list) -> AsyncMock:
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.process = AsyncMock(side_effect=process_side_effects)
    return mock_session


@pytest.mark.asyncio
async def test_voice_loop_routes_command_to_session() -> None:
    mock_session = _make_session_mock(
        ["Ich schalte den Fernseher ein.", KeyboardInterrupt()]
    )

    mock_ww = MagicMock()
    mock_ww.listen = AsyncMock()

    mock_stt = MagicMock()
    mock_stt.record_and_transcribe = AsyncMock(
        return_value=RecognitionResult(text="schalte den fernseher ein")
    )

    mock_tts = MagicMock()
    mock_tts.speak = AsyncMock()

    mock_cfg = MagicMock()

    with (
        patch("appkit_commons.registry.service_registry") as mock_registry,
        patch("pantau.audio.wakeword.WakeWordListener", return_value=mock_ww),
        patch("pantau.audio.stt.create_stt", return_value=mock_stt),
        patch("pantau.audio.tts.PiperTTS", return_value=mock_tts),
        patch("pantau.session.PantauSession", return_value=mock_session),
    ):
        mock_registry.return_value.get.return_value = mock_cfg

        from pantau.main import _voice_loop

        await _voice_loop()

    mock_session.process.assert_called()
    spoken = [call.args[0] for call in mock_tts.speak.call_args_list]
    assert "Pantau ist bereit." in spoken
    assert "Ja?" in spoken
    assert "Ich schalte den Fernseher ein." in spoken


@pytest.mark.asyncio
async def test_voice_loop_speaks_error_on_process_exception() -> None:
    mock_session = _make_session_mock([RuntimeError("MCP down"), KeyboardInterrupt()])

    mock_ww = MagicMock()
    mock_ww.listen = AsyncMock()

    mock_stt = MagicMock()
    mock_stt.record_and_transcribe = AsyncMock(
        return_value=RecognitionResult(text="licht an")
    )

    mock_tts = MagicMock()
    mock_tts.speak = AsyncMock()

    mock_cfg = MagicMock()

    with (
        patch("appkit_commons.registry.service_registry") as mock_registry,
        patch("pantau.audio.wakeword.WakeWordListener", return_value=mock_ww),
        patch("pantau.audio.stt.create_stt", return_value=mock_stt),
        patch("pantau.audio.tts.PiperTTS", return_value=mock_tts),
        patch("pantau.session.PantauSession", return_value=mock_session),
    ):
        mock_registry.return_value.get.return_value = mock_cfg

        from pantau.main import _voice_loop

        await _voice_loop()

    spoken = [call.args[0] for call in mock_tts.speak.call_args_list]
    assert any("Fehler" in s for s in spoken)


@pytest.mark.asyncio
async def test_voice_loop_skips_empty_transcription() -> None:
    mock_session = _make_session_mock([])

    mock_ww = MagicMock()
    mock_ww.listen = AsyncMock(side_effect=[None, KeyboardInterrupt()])

    mock_stt = MagicMock()
    mock_stt.record_and_transcribe = AsyncMock(return_value=RecognitionResult(text=""))

    mock_tts = MagicMock()
    mock_tts.speak = AsyncMock()

    mock_cfg = MagicMock()

    with (
        patch("appkit_commons.registry.service_registry") as mock_registry,
        patch("pantau.audio.wakeword.WakeWordListener", return_value=mock_ww),
        patch("pantau.audio.stt.create_stt", return_value=mock_stt),
        patch("pantau.audio.tts.PiperTTS", return_value=mock_tts),
        patch("pantau.session.PantauSession", return_value=mock_session),
    ):
        mock_registry.return_value.get.return_value = mock_cfg

        from pantau.main import _voice_loop

        await _voice_loop()

    mock_session.process.assert_not_called()


@pytest.mark.asyncio
async def test_voice_loop_stops_on_beende_dich() -> None:
    mock_session = _make_session_mock([])

    mock_ww = MagicMock()
    mock_ww.listen = AsyncMock()
    mock_ww.stop = MagicMock()

    mock_stt = MagicMock()
    mock_stt.record_and_transcribe = AsyncMock(
        return_value=RecognitionResult(text="beende dich")
    )

    mock_tts = MagicMock()
    mock_tts.speak = AsyncMock()

    mock_cfg = MagicMock()

    with (
        patch("appkit_commons.registry.service_registry") as mock_registry,
        patch("pantau.audio.wakeword.WakeWordListener", return_value=mock_ww),
        patch("pantau.audio.stt.create_stt", return_value=mock_stt),
        patch("pantau.audio.tts.PiperTTS", return_value=mock_tts),
        patch("pantau.session.PantauSession", return_value=mock_session),
    ):
        mock_registry.return_value.get.return_value = mock_cfg

        from pantau.main import _voice_loop

        await _voice_loop()

    mock_session.process.assert_not_called()
    spoken = [call.args[0] for call in mock_tts.speak.call_args_list]
    assert any("Wiedersehen" in s for s in spoken)
