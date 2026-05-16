from __future__ import annotations

import logging
from functools import lru_cache

from appkit_commons.configuration import BaseConfig
from appkit_commons.configuration.configuration import (
    Configuration,
    Environment,
)
from appkit_commons.registry import service_registry

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
    api_key: str = "secret:openai_api_key"
    base_url: str | None = None


class ApplicationConfig(BaseConfig):
    version: str
    name: str
    logging: str
    environment: Environment | None = Environment.local
    wake_word: WakeWordConfig = WakeWordConfig()
    stt: SttConfig = SttConfig()
    tts: TtsConfig = TtsConfig()
    llm: LlmConfig = LlmConfig()


@lru_cache(maxsize=1)
def configure() -> Configuration[ApplicationConfig]:
    logger.debug("--- Configuring application settings ---")
    return service_registry().configure(
        ApplicationConfig,
        env_file="/.env",
    )
