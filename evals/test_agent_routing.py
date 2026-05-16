"""Integration eval: agent routes German commands to correct pantau_* tools.

Requires OPENAI_API_KEY. Skip with: pytest --ignore=evals/ or -m "not integration".
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from pantau.config import load_config

COMMANDS_PATH = Path(__file__).parent / "commands_de.yaml"


def _load_cases() -> list[dict[str, Any]]:
    return yaml.safe_load(COMMANDS_PATH.read_text(encoding="utf-8"))


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize("case", _load_cases(), ids=[c["input"] for c in _load_cases()])
async def test_agent_routing(case: dict[str, Any]) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio

    cfg = load_config()
    home_mcp = MCPServerStdio("python", args=["-m", "pantau.home_mcp.server"], timeout=15)
    agent = Agent(f"openai:{cfg.llm.model}", toolsets=[home_mcp])

    tool_calls: list[str] = []

    async with agent.run_mcp_servers():
        result = await agent.run(
            case["input"],
            message_history=[],
        )

    called_tools = [
        msg.tool_name
        for msg in result.all_messages()
        if hasattr(msg, "tool_name")
    ]

    expected_tool: str = case["expected_tool"]
    assert expected_tool in called_tools, (
        f"Expected tool '{expected_tool}' not called. Called: {called_tools}"
    )

    if "expected_args" in case:
        for msg in result.all_messages():
            if getattr(msg, "tool_name", None) == expected_tool:
                args = getattr(msg, "args", {})
                for key, val in case["expected_args"].items():
                    assert str(args.get(key, "")).lower() == str(val).lower(), (
                        f"Arg '{key}': expected '{val}', got '{args.get(key)}'"
                    )
                break
    _ = tool_calls
