from __future__ import annotations

import logging
from functools import lru_cache

from appkit_commons.configuration import BaseConfig
from appkit_commons.configuration.configuration import (
    Configuration,
    Environment,
)
from appkit_commons.registry import service_registry
from pydantic import Field

logger = logging.getLogger(__name__)


class SttConfig(BaseConfig):
    model_size: str = "small"
    device: str = "auto"
    language: str = "de"


class TtsConfig(BaseConfig):
    model: str = "models/de_DE-thorsten-high.onnx"
    speak_rate: float = 1.0


class WakeWordConfig(BaseConfig):
    model: str = "models/pantau.tflite"
    threshold: float = 0.5
    post_wake_timeout_s: float = 6.0


class LlmConfig(BaseConfig):
    provider: str = "openai"
    model: str = "gpt-5.4-nano"
    api_key: str = ""
    base_url: str | None = None


class McpServerConfig(BaseConfig):
    name: str
    command: str | None = None
    args: list[str] = []
    cwd: str | None = None


class McpConfig(BaseConfig):
    servers: list[McpServerConfig] = Field(
        default_factory=lambda: [
            McpServerConfig(name="harmonyhub", args=["-m", "harmonyhub.mcp_server"]),
            McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"]),
        ]
    )


class ApplicationConfig(BaseConfig):
    version: str
    name: str
    logging: str
    environment: Environment | None = Environment.local

    wake_word: WakeWordConfig = WakeWordConfig()
    stt: SttConfig = SttConfig()
    tts: TtsConfig = TtsConfig()
    llm: LlmConfig = LlmConfig()
    mcp: McpConfig = Field(default_factory=McpConfig)


@lru_cache(maxsize=1)
def configure() -> Configuration[ApplicationConfig]:
    logger.debug("--- Configuring application settings ---")
    return service_registry().configure(
        ApplicationConfig,
        env_file=".env",
    )
