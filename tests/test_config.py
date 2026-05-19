from __future__ import annotations

from pantau.config import (
    ApplicationConfig,
    LlmConfig,
    McpConfig,
    McpServerConfig,
    PantauConfig,
    SttConfig,
    TtsConfig,
    WakeWordConfig,
    load_config,
)


def test_application_config_defaults() -> None:
    cfg = ApplicationConfig(version="0.1.0", name="pantau", logging="logging.yaml")

    assert cfg.llm.model == "gpt-5.4-nano"
    assert cfg.llm.provider == "openai"
    assert [server.name for server in cfg.mcp.servers] == ["harmonyhub", "huehub"]
    assert cfg.mcp.servers[0].command is None
    assert cfg.mcp.servers[0].args == ["-m", "harmonyhub.mcp_server"]
    assert cfg.mcp.servers[1].command is None
    assert cfg.mcp.servers[1].args == ["-m", "huehub.mcp_server"]
    assert cfg.stt.model_size == "small"
    assert cfg.stt.language == "de"
    assert cfg.tts.speak_rate == 1.0
    assert cfg.wake_word.threshold == 0.5


def test_application_config_override() -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(provider="ollama", model="qwen3", api_key=""),
    )

    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "qwen3"


def test_mcp_config_fields() -> None:
    cfg = McpConfig(
        servers=[
            McpServerConfig(
                name="code-reasoning",
                command="npx",
                args=["-y", "@mettamatt/code-reasoning"],
                cwd="/tmp/code-reasoning",  # noqa: S108
            ),
            McpServerConfig(
                name="duckduckgo",
                command="uvx",
                args=["duckduckgo-mcp-server"],
            ),
        ]
    )

    assert cfg.servers[0].name == "code-reasoning"
    assert cfg.servers[0].command == "npx"
    assert cfg.servers[0].args == ["-y", "@mettamatt/code-reasoning"]
    assert cfg.servers[0].cwd == "/tmp/code-reasoning"  # noqa: S108
    assert cfg.servers[1].name == "duckduckgo"
    assert cfg.servers[1].command == "uvx"
    assert cfg.servers[1].args == ["duckduckgo-mcp-server"]


def test_llm_config_fields() -> None:
    cfg = LlmConfig(provider="ollama", model="qwen3", api_key="test")
    assert cfg.provider == "ollama"
    assert cfg.model == "qwen3"


def test_stt_config_defaults() -> None:
    cfg = SttConfig()
    assert cfg.model_size == "small"
    assert cfg.device == "auto"
    assert cfg.language == "de"


def test_wake_word_config_defaults() -> None:
    cfg = WakeWordConfig()
    assert cfg.threshold == 0.5
    assert cfg.post_wake_timeout_s == 3.0


def test_tts_config_defaults() -> None:
    cfg = TtsConfig()
    assert cfg.model == "models/de_DE-thorsten-high.onnx"
    assert cfg.speak_rate == 1.0


def test_pantau_config_is_alias_for_application_config() -> None:
    assert PantauConfig is ApplicationConfig


def test_load_config_returns_application_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, ApplicationConfig)
    assert cfg.llm.model
