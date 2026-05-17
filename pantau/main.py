from __future__ import annotations

import asyncio
import logging
import string
import sys

logger = logging.getLogger(__name__)

_STOP_PHRASES = frozenset({"beende dich", "auf wiedersehen", "exit"})


def _is_stop_phrase(text: str) -> bool:
    return text.strip().lower().rstrip(string.punctuation) in _STOP_PHRASES


async def _repl() -> None:
    from pantau.session import PantauSession, process

    logger.info("Pantau text agent ready. Type a command (Ctrl-C to exit).")
    async with PantauSession() as session:
        while True:
            try:
                text = await asyncio.to_thread(input, "> ")
                text = text.strip()
            except EOFError, KeyboardInterrupt:
                break

            if not text:
                continue
            if _is_stop_phrase(text):
                break
            try:
                response = await process(text, session=session)
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
            except Exception:
                logger.exception("Error processing command")


async def _voice_loop() -> None:
    from appkit_commons.registry import service_registry

    from pantau.audio.stt import create_stt
    from pantau.audio.tts import PiperTTS
    from pantau.audio.wakeword import WakeWordListener
    from pantau.config import ApplicationConfig
    from pantau.session import PantauSession

    cfg = service_registry().get(ApplicationConfig)
    logger.debug("Starting voice loop with configuration: %s", cfg)
    wakeword = WakeWordListener(cfg.wake_word)
    stt = create_stt(cfg.stt)
    tts = PiperTTS(cfg.tts)

    async with PantauSession(cfg) as session:
        await tts.speak("Pantau ist bereit.")
        logger.info("Voice loop ready; waiting for wake word...")

        while True:
            try:
                await wakeword.listen()
                logger.debug("Wake word detected, entering follow-up loop")
                await tts.speak("Ja?")

                while True:
                    result = await stt.record_and_transcribe(
                        initial_silence_timeout_s=cfg.wake_word.post_wake_timeout_s,
                    )
                    logger.debug("Transcription complete: %s", result.text)
                    if not result.text:
                        logger.debug(
                            "No speech detected, returning to wake word listening"
                        )
                        break

                    if result.intent == "stop_session" or _is_stop_phrase(result.text):
                        logger.info("Stop phrase detected, shutting down")
                        await tts.speak("Auf Wiedersehen.")
                        return

                    response = await session.process(result.text)
                    logger.debug("Response: %s", response)
                    await tts.speak(response)

            except KeyboardInterrupt:
                logger.debug("Voice loop stopped by user")
                break
            except Exception:
                logger.exception("Unhandled error in voice loop")
                await tts.speak("Es gab einen Fehler. Bitte versuche es erneut.")


def main() -> None:
    if "--voice" in sys.argv:
        asyncio.run(_voice_loop())
    else:
        asyncio.run(_repl())


if __name__ == "__main__":
    main()
