from pantau.agent.runtime import build_agent
from pantau.config import ApplicationConfig, LlmConfig


def inspect(name: str, cfg: ApplicationConfig) -> None:
    try:
        agent = build_agent(cfg)
        for attr in ["model", "_model", "model_settings"]:
            hasattr(agent, attr)
            getattr(agent, attr, "N/A")

        # Look for model object
        model_obj = None
        if hasattr(agent, "model"):
            model_obj = agent.model
        elif hasattr(agent, "_model"):
            model_obj = agent._model  # noqa: SLF001

        if model_obj:
            pass

        [a for a in dir(agent) if not a.startswith("__")]
    except Exception:
        import traceback

        traceback.print_exc()


base_cfg = {"version": "1.0", "name": "test", "logging": "INFO"}

# OpenAI
openai_cfg = ApplicationConfig(
    **base_cfg, llm=LlmConfig(provider="openai", model="gpt-4o")
)
inspect("OpenAI", openai_cfg)

# Ollama
ollama_cfg = ApplicationConfig(
    **base_cfg, llm=LlmConfig(provider="ollama", model="llama3")
)
inspect("Ollama", ollama_cfg)
