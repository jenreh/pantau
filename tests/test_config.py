from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pantau.config import LlmConfig, PantauConfig, load_config


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "pantau.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def test_load_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    _write_yaml(tmp_path / "config", {"log_level": "DEBUG"})

    cfg = load_config()

    assert cfg.log_level == "DEBUG"
    assert cfg.llm.model == "gpt-5.4-nano"
    assert cfg.harmony.host == "192.168.1.50"


def test_load_config_yaml_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    _write_yaml(
        tmp_path / "config",
        {
            "llm": {"provider": "ollama", "model": "qwen3"},
            "harmony": {"host": "10.0.0.99"},
        },
    )

    cfg = load_config()

    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "qwen3"
    assert cfg.harmony.host == "10.0.0.99"


def test_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    _write_yaml(tmp_path / "config", {})
    monkeypatch.setenv("PANTAU_LLM__MODEL", "gpt-override")

    cfg = load_config()

    assert cfg.llm.model == "gpt-override"


def test_llm_config_fields() -> None:
    cfg = LlmConfig(provider="ollama", model="qwen3", api_key="test")
    assert cfg.provider == "ollama"
    assert cfg.model == "qwen3"


def test_pantau_config_env_prefix() -> None:
    assert PantauConfig.model_config.get("env_prefix") == "PANTAU_"
    assert PantauConfig.model_config.get("env_nested_delimiter") == "__"
