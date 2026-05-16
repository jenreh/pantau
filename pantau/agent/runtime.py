from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from pantau.agent.prompts import SYSTEM_PROMPT
from pantau.config import PantauConfig


def build_agent(cfg: PantauConfig) -> Agent:
    model_str = (
        f"ollama:{cfg.llm.model}"
        if cfg.llm.provider == "ollama"
        else f"openai:{cfg.llm.model}"
    )

    home_mcp = MCPServerStdio(  # FIXME: MCPServerStdio is deprecated!
        "python",
        args=["-m", "pantau.home_mcp.server"],
        timeout=10,
    )
    return Agent(
        model_str,
        instructions=SYSTEM_PROMPT,
        toolsets=[home_mcp],
    )
