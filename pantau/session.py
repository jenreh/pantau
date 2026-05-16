from __future__ import annotations

import logging
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
from pantau.config import ApplicationConfig

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
        self.agent: Agent[Any, Any] | None = None
        self.available_mcp_servers = []
        self.message_history: list[ModelMessage] = []
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self) -> PantauSession:
        if self.cfg is None:
            self.cfg = service_registry().get(ApplicationConfig)

        self.available_mcp_servers = await resolve_available_mcp_servers(
            self.cfg.mcp.servers,
        )
        self.agent = build_agent(self.cfg, mcp_servers=self.available_mcp_servers)
        await self._exit_stack.enter_async_context(self.agent)
        logger.info(
            "pantau-session: initialized agent with %d available MCP server(s)",
            len(self.available_mcp_servers),
        )
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self._exit_stack.aclose()

    async def process(self, text: str) -> str:
        if self.cfg is None or self.agent is None:
            raise RuntimeError("PantauSession must be entered before use")

        fast_path_response = await _process_fast_path(self.cfg, text)
        if fast_path_response is not None:
            _append_fast_path_history(self.message_history, text, fast_path_response)
            return fast_path_response

        logger.info(
            "llm-path: routing to session agent with %d available MCP server(s)",
            len(self.available_mcp_servers),
        )
        result = await self.agent.run(text, message_history=self.message_history)
        self.message_history = result.all_messages()
        return str(result.output)


async def process(text: str, session: PantauSession | None = None) -> str:
    if session is not None:
        return await session.process(text)

    cfg = service_registry().get(ApplicationConfig)
    fast_path_response = await _process_fast_path(cfg, text)
    if fast_path_response is not None:
        return fast_path_response

    available_mcp_servers = await resolve_available_mcp_servers(cfg.mcp.servers)
    agent = build_agent(cfg, mcp_servers=available_mcp_servers)
    logger.info(
        "llm-path: routing to agent with %d available MCP server(s)",
        len(available_mcp_servers),
    )
    async with agent:
        result = await agent.run(text)
    return str(result.output)
