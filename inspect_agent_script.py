from pantau.agent.runtime import build_agent
from pantau.config import AgentConfig, ApplicationConfig, ToolsetConfig

# Minimal config to build an agent
agent_config = AgentConfig(
    name="test-agent",
    llm={"provider": "openai", "model": "gpt-4o"},
    toolsets=[ToolsetConfig(type="http_mcp", url="http://localhost:8000")],
)
cfg = ApplicationConfig(agent=agent_config)

try:
    agent = build_agent(cfg)
    if hasattr(agent, "toolsets") and agent.toolsets:
        first_toolset = agent.toolsets[0]
        if hasattr(first_toolset, "url"):
            pass
except Exception:
    import traceback

    traceback.print_exc()
