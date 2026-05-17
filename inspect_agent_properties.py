from typing import Any

from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider


def inspect(name: str, model: Any) -> None:
    if hasattr(model, "provider"):
        p = model.provider
        # Check for nested client/_client
        client = getattr(p, "client", getattr(p, "_client", None))
        if client:
            pass
        else:
            pass


ollama_m = OllamaModel(
    "llama3",
    provider=OllamaProvider(base_url="http://localhost:11434", api_key="ollama-key"),
)
inspect("Ollama", ollama_m)

openai_m = OpenAIResponsesModel(
    "gpt-4",
    provider=OpenAIProvider(base_url="https://api.openai.com/v1", api_key="sk-test"),
)
inspect("OpenAI", openai_m)
