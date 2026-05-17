# Pantau

![Version](https://img.shields.io/badge/version-1.1.0-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.14%2B-orange)](https://www.python.org)

Local voice agent for smart home control — German commands, runs on your LAN.

Say **"Pantau"** → issue a command → done.

```text
Microphone → Wake Word → VAD → STT → Fast Path / LLM Agent → TTS → Speaker
                                                 → (Skills)
                                                 → MCP
                                                 → Smart Home
```

---

## What it does

- Listens for the wake word **"Pantau"** using [OpenWakeWord](https://github.com/dscripka/openWakeWord)
- Transcribes German speech locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Routes commands through a fast path (deterministic, no LLM) or a [pydantic-ai](https://ai.pydantic.dev) agent
- Controls lights, TV, music, and blinds via [FastMCP](https://github.com/jlowin/fastmcp) — directly over your LAN
- Responds in German using [Piper TTS](https://github.com/rhasspy/piper) (local, no cloud)

Internet is required only for the LLM in the default config. Everything else is local.

### Smart home integrations

| Device | MCP server |
| --- | --- |
| Philips Hue lights | [huehub-py](https://github.com/jenreh/huehub-py) |
| Logitech Harmony Hub (TV) | [harmonyhub-py](https://github.com/jenreh/harmonyhub-py) |
| Sonos speakers | [sonos-py](https://github.com/jenreh/sonos-py) |
| HomeKit blinds/covers | [homekit-py](https://github.com/jenreh/homekit-py) |

---

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- [`task`](https://taskfile.dev/installation/) — task runner
- A working microphone and speaker
- An OpenAI API key (for the default LLM; see [Switching to local LLM](#switching-to-a-local-llm) to remove this requirement)

---

## Setup

### 1. Install dependencies

```sh
task init
```

Installs the required Python version, syncs dependencies, and sets up pre-commit hooks.

### 2. Download models

```sh
task download-models
```

Downloads the Piper TTS voice (`de_DE-thorsten-high`) and the OpenWakeWord pretrained models.

> [!NOTE]
> To use the custom **"Pantau"** wake word, place your trained `pantau.tflite` model in the `models/` directory.
> Until then, the agent falls back to OpenWakeWord's pretrained models (e.g., "hey jarvis").
> Train a custom model using the [OpenWakeWord training pipeline](https://github.com/dscripka/openWakeWord/blob/main/docs/training.md).

### 3. Configure smart home servers

Each MCP server must be configured before first use via its own CLI. This stores your device credentials locally — Pantau itself never touches them directly.

| Server | Documentation |
| --- | --- |
| **huehub-py** (Hue lights) | [huehub-py docs](https://github.com/jenreh/huehub-py#configuration) |
| **harmonyhub-py** (Harmony Hub) | [harmonyhub-py docs](https://github.com/jenreh/harmonyhub-py#configuration) |
| **sonos-py** (Sonos speakers) | [sonos-py docs](https://github.com/jenreh/sonos-py#configuration) |
| **homekit-py** (HomeKit blinds) | [homekit-py docs](https://github.com/jenreh/homekit-py#configuration) |

> [!IMPORTANT]
> Pantau will skip any MCP server whose CLI command is not found on `PATH`. Run the setup commands above before starting Pantau, or the corresponding devices will not respond.

### 4. Set your OpenAI API key

Create a `.env` file in the project root:

```sh
OPENAI_API_KEY=sk-...
```

---

## Running

```sh
# Text REPL (useful for testing without audio hardware)
pantau

# Voice mode
pantau --voice
```

Stop with **"Beende dich"**, **"Auf Wiedersehen"**, or **Ctrl-C**.

---

## Configuration

Base config lives in `configuration/config.yaml`. Override any value with environment variables using `PANTAU_` prefix and `__` as the nested separator.

```yaml
wake_word:
  model: models/pantau.tflite   # path to your .tflite wake word model
  threshold: 0.5                # detection sensitivity (0–1)
  post_wake_timeout_s: 6.0      # seconds to wait for speech after activation

stt:
  model_size: medium            # tiny | base | small | medium | large-v3
  device: auto                  # auto | cpu | cuda
  language: de

llm:
  provider: openai              # openai | ollama
  model: gpt-5.4-nano
  api_key: secret:pt-openai-api-key   # resolved from env OPENAI_API_KEY
```

### Switching to a local LLM

```sh
export PANTAU_LLM__PROVIDER=ollama
export PANTAU_LLM__MODEL=qwen3
export PANTAU_LLM__BASE_URL=http://localhost:11434/v1
```

Requires [Ollama](https://ollama.com) running locally with `qwen3` pulled.

---

## Latency reference

| Stage | Typical |
| --- | --- |
| Wake word detection | < 100 ms |
| STT (`medium`, CPU) | ~1 s per 3 s of audio |
| Fast path (common phrases) | < 1 ms |
| LLM round-trip (`gpt-5.4-nano`) | 300–700 ms |
| TTS synthesis + playback | 200–400 ms |
| **Total — fast path** | **~2–3 s** |
| **Total — LLM path** | **~3–5 s** |

---

## Development

```sh
task test      # run tests with coverage
task lint      # ruff lint + format check
task format    # auto-format
```
