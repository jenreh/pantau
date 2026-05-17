"""Integration eval: agent routes German commands to correct MCP tools.

Requires OPENAI_API_KEY and running MCP servers. Skip with:
  pytest --ignore=evals/
  pytest -m "not integration"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from pantau.agent.runtime import build_agent, resolve_available_mcp_servers
from pantau.config import ApplicationConfig, LlmConfig, McpConfig, McpServerConfig

COMMANDS_PATH = Path(__file__).parent / "commands_de.yaml"


def _load_cases() -> list[dict[str, Any]]:
    return yaml.safe_load(COMMANDS_PATH.read_text(encoding="utf-8"))


def _default_eval_cfg() -> ApplicationConfig:
    return ApplicationConfig(
        version="0.1.0",
        name="pantau-eval",
        logging="logging.yaml",
        llm=LlmConfig(
            provider="openai",
            model="gpt-5.4-nano",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
        mcp=McpConfig(
            servers=[
                McpServerConfig(
                    name="harmonyhub", args=["-m", "harmonyhub.mcp_server"]
                ),
                McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"]),
            ]
        ),
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize("case", _load_cases(), ids=[c["input"] for c in _load_cases()])
async def test_agent_routing(case: dict[str, Any]) -> None:
    from pydantic_ai.messages import ToolCallPart

    cfg = _default_eval_cfg()
    available_servers = await resolve_available_mcp_servers(cfg.mcp.servers)
    agent = build_agent(cfg, mcp_servers=available_servers)

    async with agent:
        result = await agent.run(case["input"], message_history=[])

    tool_calls: list[ToolCallPart] = [
        part
        for msg in result.all_messages()
        for part in getattr(msg, "parts", [])
        if isinstance(part, ToolCallPart)
    ]
    called_names = [tc.tool_name for tc in tool_calls]

    expected_tool: str = case["expected_tool"]
    assert expected_tool in called_names, (
        f"Expected tool '{expected_tool}' not called. Called: {called_names}"
    )

    if "expected_args" in case:
        for tc in tool_calls:
            if tc.tool_name != expected_tool:
                continue
            args = tc.args_as_dict()
            for key, val in case["expected_args"].items():
                assert str(args.get(key, "")).lower() == str(val).lower(), (
                    f"Arg '{key}': expected '{val}', got '{args.get(key)}'"
                )
            break
