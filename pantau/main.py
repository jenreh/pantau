from __future__ import annotations

import asyncio
import logging
import sys
from importlib import import_module

logger = logging.getLogger(__name__)


async def _repl() -> None:
    session_module = import_module("pantau.session")
    pantau_session = session_module.PantauSession
    process_text = session_module.process

    logger.info("Pantau text agent ready. Type a command (Ctrl-C to exit).")
    async with pantau_session() as session:
        while True:
            try:
                text = await asyncio.to_thread(input, "> ")
                text = text.strip()
            except EOFError, KeyboardInterrupt:
                break

            if not text:
                continue
            try:
                response = await process_text(text, session=session)
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
            except Exception:
                logger.exception("Error processing command")


def main() -> None:
    asyncio.run(_repl())


if __name__ == "__main__":
    main()
