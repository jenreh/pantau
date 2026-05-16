from __future__ import annotations

import contextlib
import sys
from unittest.mock import MagicMock

import pytest
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from pantau.agent import runtime
from pantau.agent.runtime import (
    _call_tool_on_server,
    build_agent,
    execute_mcp_tool,
    resolve_available_mcp_servers,
)
from pantau.config import ApplicationConfig, LlmConfig, McpConfig, McpServerConfig


def test_build_agent_uses_openai_responses_model_and_provider_config() -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(
            provider="openai",
            model="gpt-5.4-nano",
            api_key="test-openai-key",
            base_url="https://example.invalid/v1",
        ),
        mcp=McpConfig(
            servers=[
                McpServerConfig(
                    name="harmonyhub", args=["-m", "harmonyhub.mcp_server"]
                ),
                McpServerConfig(
                    name="duckduckgo",
                    command="uvx",
                    args=["duckduckgo-mcp-server"],
                ),
            ],
        ),
    )

    agent = build_agent(cfg)

    assert isinstance(agent.model, OpenAIResponsesModel)
    assert agent.model.model_name == "gpt-5.4-nano"
    assert str(agent.model.provider.base_url) == "https://example.invalid/v1/"
    assert agent.model.provider.client.api_key == "test-openai-key"

    mcp_toolsets = [
        toolset for toolset in agent.toolsets if isinstance(toolset, MCPToolset)
    ]

    assert len(mcp_toolsets) == 2
    assert mcp_toolsets[0].client.transport.command == sys.executable
    assert mcp_toolsets[0].client.transport.args == ["-m", "harmonyhub.mcp_server"]
    assert mcp_toolsets[1].client.transport.command == "uvx"
    assert mcp_toolsets[1].client.transport.args == ["duckduckgo-mcp-server"]


def test_build_agent_uses_ollama_provider_config() -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(
            provider="ollama",
            model="qwen3",
            api_key="ollama-test-key",
            base_url="http://localhost:11434/v1",
        ),
        mcp=McpConfig(
            servers=[
                McpServerConfig(
                    name="code-reasoning",
                    command="npx",
                    args=["-y", "@mettamatt/code-reasoning"],
                    cwd="/tmp/code-reasoning",  # noqa: S108
                ),
                McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"]),
            ],
        ),
    )

    agent = build_agent(cfg)

    assert isinstance(agent.model, OllamaModel)
    assert agent.model.model_name == "qwen3"
    assert str(agent.model.provider.base_url) == "http://localhost:11434/v1/"
    assert agent.model.provider.client.api_key == "ollama-test-key"

    mcp_toolsets = [
        toolset for toolset in agent.toolsets if isinstance(toolset, MCPToolset)
    ]

    assert len(mcp_toolsets) == 2
    assert mcp_toolsets[0].client.transport.command == "npx"
    assert mcp_toolsets[0].client.transport.args == ["-y", "@mettamatt/code-reasoning"]
    assert mcp_toolsets[0].client.transport.cwd == "/tmp/code-reasoning"  # noqa: S108
    assert mcp_toolsets[1].client.transport.command == sys.executable
    assert mcp_toolsets[1].client.transport.args == ["-m", "huehub.mcp_server"]


@pytest.mark.asyncio
async def test_resolve_available_mcp_servers_skips_unavailable_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(),
        mcp=McpConfig(
            servers=[
                McpServerConfig(name="broken", args=["-m", "broken.server"]),
                McpServerConfig(name="working", args=["-m", "working.server"]),
            ],
        ),
    )

    class FakeToolset:
        def __init__(self, server_name: str) -> None:
            self.server_name = server_name

        async def __aenter__(self) -> FakeToolset:
            if self.server_name == "broken":
                raise RuntimeError("Connection closed")
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    monkeypatch.setattr(
        runtime,
        "_build_stdio_mcp_toolset",
        lambda server: FakeToolset(server.name),
    )

    available_servers = await resolve_available_mcp_servers(cfg.mcp.servers)

    assert [server.name for server in available_servers] == ["working"]


def test_build_agent_can_use_filtered_mcp_servers() -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(),
        mcp=McpConfig(
            servers=[
                McpServerConfig(
                    name="harmonyhub", args=["-m", "harmonyhub.mcp_server"]
                ),
                McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"]),
            ],
        ),
    )

    agent = build_agent(cfg, mcp_servers=[cfg.mcp.servers[1]])

    mcp_toolsets = [
        toolset for toolset in agent.toolsets if isinstance(toolset, MCPToolset)
    ]

    assert len(mcp_toolsets) == 1
    assert mcp_toolsets[0].client.transport.args == ["-m", "huehub.mcp_server"]


@pytest.mark.asyncio
async def test_execute_mcp_tool_uses_first_server_that_exposes_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers = [
        McpServerConfig(name="harmonyhub", args=["-m", "harmonyhub.mcp_server"]),
        McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"]),
    ]
    calls: list[tuple[str, str, dict[str, object]]] = []

    async def fake_call_tool_on_server(
        server: McpServerConfig,
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object] | None:
        calls.append((server.name, tool_name, arguments))
        if server.name == "harmonyhub":
            return None
        return {"status": "ok"}

    monkeypatch.setattr(runtime, "_call_tool_on_server", fake_call_tool_on_server)

    result = await execute_mcp_tool(
        servers,
        "hue_set_room_on",
        {"room": "Flur", "on": True},
    )

    assert result == {"status": "ok"}
    assert calls == [
        ("harmonyhub", "hue_set_room_on", {"room": "Flur", "on": True}),
        ("huehub", "hue_set_room_on", {"room": "Flur", "on": True}),
    ]


@pytest.mark.asyncio
async def test_execute_mcp_tool_raises_when_tool_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers = [McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"])]

    async def fake_call_tool_on_server(
        server: McpServerConfig,
        tool_name: str,
        arguments: dict[str, object],
    ) -> None:
        return None

    monkeypatch.setattr(runtime, "_call_tool_on_server", fake_call_tool_on_server)

    with pytest.raises(LookupError, match="hue_set_room_on"):
        await execute_mcp_tool(servers, "hue_set_room_on", {"room": "Flur"})


@pytest.mark.asyncio
async def test_execute_mcp_tool_raises_runtime_error_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers = [McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"])]

    async def fake_call_tool_on_server(
        server: McpServerConfig,
        tool_name: str,
        arguments: dict[str, object],
    ) -> None:
        raise ValueError("Simulated network failure")

    monkeypatch.setattr(runtime, "_call_tool_on_server", fake_call_tool_on_server)

    with pytest.raises(RuntimeError, match="Unable to execute MCP tool"):
        await execute_mcp_tool(servers, "hue_set_room_on", {"room": "Flur"})


@pytest.mark.asyncio
async def test_call_tool_on_server_returns_result_when_tool_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"])

    @contextlib.asynccontextmanager
    async def fake_stdio_client(*args, **kwargs):
        yield "fake_read", "fake_write"

    class FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            pass

        async def list_tools(self):
            tools_mock = MagicMock()
            tool_mock = MagicMock()
            tool_mock.name = "hue_set_room_on"
            tools_mock.tools = [tool_mock]
            return tools_mock

        async def call_tool(self, name, arguments):
            return {"status": "success", "room": arguments.get("room")}

    monkeypatch.setattr(runtime, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(runtime, "ClientSession", FakeSession)

    result = await _call_tool_on_server(server, "hue_set_room_on", {"room": "Flur"})
    assert result == {"status": "success", "room": "Flur"}


@pytest.mark.asyncio
async def test_call_tool_on_server_returns_none_when_tool_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"])

    @contextlib.asynccontextmanager
    async def fake_stdio_client(*args, **kwargs):
        yield "fake_read", "fake_write"

    class FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            pass

        async def list_tools(self):
            tools_mock = MagicMock()
            tools_mock.tools = []
            return tools_mock

    monkeypatch.setattr(runtime, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(runtime, "ClientSession", FakeSession)

    result = await _call_tool_on_server(server, "missing_tool", {})
    assert result is None
