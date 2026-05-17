from __future__ import annotations

import asyncio
import logging
import sys

logger = logging.getLogger(__name__)

_STOP_PHRASES = frozenset({"beende dich", "auf wiedersehen", "exit"})


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
            if text.lower() in _STOP_PHRASES:
                break
            try:
                response = await process(text, session=session)
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
            except Exception:
                logger.exception("Error processing command")


async def _voice_loop() -> None:
    from appkit_commons.registry import service_registry

    from pantau.audio.stt import GermanSTT
    from pantau.audio.tts import PiperTTS
    from pantau.audio.wakeword import WakeWordListener
    from pantau.config import ApplicationConfig
    from pantau.session import PantauSession

    cfg = service_registry().get(ApplicationConfig)
    wakeword = WakeWordListener(cfg.wake_word)
    stt = GermanSTT(cfg.stt)
    tts = PiperTTS(cfg.tts)

    async with PantauSession(cfg) as session:
        await tts.speak("Pantau ist bereit.")
        logger.info("Voice loop ready; waiting for wake word...")

        while True:
            try:
                await wakeword.listen()
                logger.debug("Wake word detected, starting voice interaction")
                await tts.speak("Ja?")

                text = await stt.record_and_transcribe()
                logger.debug("Transcription complete: %s", text)
                if not text:
                    wakeword.stop()
                    logger.debug("No speech detected, returning to wake word listening")
                    continue

                if text.strip().lower() in _STOP_PHRASES:
                    logger.info("Stop phrase detected, shutting down")
                    await tts.speak("Auf Wiedersehen.")
                    wakeword.stop()
                    break

                response = await session.process(text)
                logger.debug("Response: %s", response)
                await tts.speak(response)
            except KeyboardInterrupt:
                logger.debug("Voice loop stopped by user")
                wakeword.stop()
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
