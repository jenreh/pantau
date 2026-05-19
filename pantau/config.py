from __future__ import annotations

import logging
from functools import lru_cache

from appkit_commons.configuration import BaseConfig
from appkit_commons.configuration.configuration import (
    Configuration,
    Environment,
)
from appkit_commons.registry import service_registry
from pydantic import Field, SecretStr

logger = logging.getLogger(__name__)


class SttConfig(BaseConfig):
    provider: str = "faster_whisper"  # faster_whisper | mlx_whisper | vosk | picovoice | simul_streaming
    model_size: str = "small"
    model_path: str = ""  # vosk: path to model directory
    device: str = "auto"  # faster_whisper / mlx_whisper only
    language: str = "de"
    silence_stop_s: float = 0.4  # post-speech silence before cutoff (was hardcoded 1.2)
    api_key: SecretStr = SecretStr("")
    intent_enabled: bool = False
    initial_prompt: str = "Schalte das Licht ein. Schalte den Fernseher aus. Mach lauter. Wohnzimmer, Schlafzimmer, Küche."
    # simul_streaming backend
    simul_model_path: str = (
        ""  # path to whisper .pt file (required for simul_streaming)
    )
    simul_cif_ckpt_path: str = (
        ""  # optional CIF checkpoint for AlignAtt word boundary detection
    )
    simul_frame_threshold: int = 25  # AlignAtt attention threshold (frames × 0.02 s)
    simul_audio_max_len: float = 30.0  # max audio buffer in seconds
    simul_beams: int = 1  # beam width (1 = greedy)


class TtsConfig(BaseConfig):
    model: str = "models/de_DE-thorsten-high.onnx"
    speak_rate: float = 1.0


class WakeWordConfig(BaseConfig):
    model: str = "models/pantau.tflite"
    threshold: float = 0.5
    post_wake_timeout_s: float = 3.0


class LlmConfig(BaseConfig):
    provider: str = "openai"
    model: str = "gpt-5.4-nano"
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None


class McpServerConfig(BaseConfig):
    name: str
    command: str | None = None
    args: list[str] = []
    cwd: str | None = None
    init_timeout: float = 5.0


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


PantauConfig = ApplicationConfig


@lru_cache(maxsize=1)
def configure() -> Configuration[ApplicationConfig]:
    logger.debug("--- Configuring application settings ---")
    return service_registry().configure(
        ApplicationConfig,
        env_file=".env",
    )


def load_config() -> PantauConfig:
    configure()
    return service_registry().get(ApplicationConfig)
