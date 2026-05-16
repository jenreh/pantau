from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from fastmcp.client.transports import StdioTransport
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from pantau.agent.prompts import SYSTEM_PROMPT
from pantau.config import ApplicationConfig, McpServerConfig

logger = logging.getLogger(__name__)


def _build_stdio_server_parameters(server: McpServerConfig) -> StdioServerParameters:
    return StdioServerParameters(
        command=server.command or sys.executable,
        args=list(server.args),
        env=None,
    )


def _build_stdio_mcp_toolset(server: McpServerConfig) -> MCPToolset:
    return MCPToolset(
        StdioTransport(
            command=_build_stdio_server_parameters(server).command,
            args=_build_stdio_server_parameters(server).args,
            cwd=server.cwd,
        )
    )


async def _call_tool_on_server(
    server: McpServerConfig,
    tool_name: str,
    arguments: Mapping[str, object],
) -> Any | None:
    async with (
        stdio_client(_build_stdio_server_parameters(server)) as (read, write),
        ClientSession(
            read,
            write,
        ) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        if tool_name not in {tool.name for tool in tools.tools}:
            return None

        return await session.call_tool(tool_name, arguments=dict(arguments))


async def execute_mcp_tool(
    servers: Sequence[McpServerConfig],
    tool_name: str,
    arguments: Mapping[str, object],
) -> Any:
    last_error: Exception | None = None

    for server in servers:
        try:
            result = await _call_tool_on_server(server, tool_name, arguments)
            if result is None:
                continue

            logger.info(
                "Executed MCP tool '%s' on server '%s'",
                tool_name,
                server.name,
            )
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Failed MCP tool '%s' on server '%s': %s",
                tool_name,
                server.name,
                exc,
            )

    if last_error is not None:
        raise RuntimeError(f"Unable to execute MCP tool {tool_name!r}") from last_error

    raise LookupError(f"No configured MCP server exposes tool {tool_name!r}")


async def resolve_available_mcp_servers(
    servers: Sequence[McpServerConfig],
) -> list[McpServerConfig]:
    available_servers: list[McpServerConfig] = []

    for server in servers:
        try:
            async with _build_stdio_mcp_toolset(server):
                available_servers.append(server)
        except Exception as exc:
            logger.warning(
                "Skipping unavailable MCP server '%s': %s",
                server.name,
                exc,
            )

    return available_servers


def build_agent(
    cfg: ApplicationConfig,
    mcp_servers: Sequence[McpServerConfig] | None = None,
) -> Agent:
    if cfg.llm.provider == "ollama":
        model = OllamaModel(
            cfg.llm.model,
            provider=OllamaProvider(
                api_key=cfg.llm.api_key or None,
                base_url=cfg.llm.base_url,
            ),
        )
    else:
        model = OpenAIResponsesModel(
            cfg.llm.model,
            provider=OpenAIProvider(
                api_key=cfg.llm.api_key or None,
                base_url=cfg.llm.base_url,
            ),
        )

    configured_servers = mcp_servers if mcp_servers is not None else cfg.mcp.servers
    mcp_toolsets = [_build_stdio_mcp_toolset(server) for server in configured_servers]

    return Agent(
        model,
        instructions=SYSTEM_PROMPT,
        toolsets=mcp_toolsets,
    )
