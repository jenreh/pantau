from __future__ import annotations

import logging
import time
from contextlib import AsyncExitStack
from typing import Any

from appkit_commons.registry import service_registry
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from pantau.agent.fast_path import fast_path
from pantau.agent.runtime import (
    build_agent,
    execute_mcp_tool,
    resolve_available_mcp_servers,
)
from pantau.config import ApplicationConfig, McpServerConfig

logger = logging.getLogger(__name__)


async def _process_fast_path(cfg: ApplicationConfig, text: str) -> str | None:
    match = fast_path(text)
    if match is None:
        return None

    logger.info(
        "fast-path: intent=%s entity=%s tool=%s",
        match.intent_id,
        match.entity_id,
        match.tool,
    )
    await execute_mcp_tool(cfg.mcp.servers, match.tool, match.args)
    if match.response:
        return match.response
    return "Erledigt."


def _append_fast_path_history(
    message_history: list[ModelMessage],
    user_text: str,
    assistant_text: str,
) -> None:
    message_history.extend(
        [
            ModelRequest.user_text_prompt(user_text),
            ModelResponse(
                parts=[TextPart(assistant_text)], model_name="pantau-fast-path"
            ),
        ]
    )


class PantauSession:
    def __init__(self, cfg: ApplicationConfig | None = None) -> None:
        self.cfg = cfg
        self._agent: Agent[Any, Any] | None = None
        self._available_mcp_servers: list[McpServerConfig] = []
        self._message_history: list[ModelMessage] = []
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self) -> PantauSession:
        if self.cfg is None:
            self.cfg = service_registry().get(ApplicationConfig)

        self._available_mcp_servers = resolve_available_mcp_servers(
            self.cfg.mcp.servers,
        )
        self._agent = build_agent(self.cfg, mcp_servers=self._available_mcp_servers)
        await self._exit_stack.enter_async_context(self._agent)
        logger.info(
            "pantau-session: initialized agent with %d available MCP server(s)",
            len(self._available_mcp_servers),
        )
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self._exit_stack.aclose()

    async def process(self, text: str) -> str:
        if self.cfg is None or self._agent is None:
            raise RuntimeError("PantauSession must be entered before use")

        t0 = time.monotonic()
        fast_path_response = await _process_fast_path(self.cfg, text)
        if fast_path_response is not None:
            logger.info(
                "latency: fast-path=%.3fs",
                time.monotonic() - t0,
            )
            _append_fast_path_history(self._message_history, text, fast_path_response)
            return fast_path_response

        logger.info(
            "llm-path: routing to session agent with %d available MCP server(s)",
            len(self._available_mcp_servers),
        )
        result = await self._agent.run(text, message_history=self._message_history)
        logger.info(
            "latency: llm-path=%.3fs",
            time.monotonic() - t0,
        )
        self._message_history = result.all_messages()
        return str(result.output)


async def process(text: str, session: PantauSession | None = None) -> str:
    if session is not None:
        return await session.process(text)
    async with PantauSession() as s:
        return await s.process(text)
