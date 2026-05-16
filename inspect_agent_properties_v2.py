from typing import Any

from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider


def check(obj: Any, path: str) -> str:
    parts = path.split(".")
    curr = obj
    for p in parts:
        if hasattr(curr, p):
            curr = getattr(curr, p)
        else:
            return f"{path} - MISSING at {p}"
    return f"{path} - OK: {curr}"


ollama_m = OllamaModel(
    "llama3", provider=OllamaProvider(base_url="http://localhost:11434", api_key="okey")
)

openai_m = OpenAIResponsesModel(
    "gpt-4",
    provider=OpenAIProvider(base_url="https://api.openai.com/v1", api_key="sk-test"),
)
