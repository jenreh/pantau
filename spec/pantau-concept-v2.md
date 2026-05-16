# Pantau — Voice Agent: Implementation Concept v3

---

## Architecture

The full pipeline is shown in the interactive architecture diagram rendered earlier in this
conversation (SVG, clickable nodes). Pipeline in sequence:

```
Microphone
  → Wake word detector  (OpenWakeWord · "Pantau")
  → VAD / endpoint      (Silero VAD)
  → STT local           (faster-whisper · language=de)
  → Fast-path router    (deterministic, bypasses LLM for common phrases)
  → Pydantic AI agent   (gpt-5.4-nano  →  Ollama qwen3 offline target)
  → FastMCP facade      (pantau-home · 7 high-level tools)
      ├─ Harmony MCP    (harmonyhub-py  · TV / activities)
      ├─ Hue MCP        (huehub-py      · lights / scenes / rooms)
      ├─ Sonos MCP      (sonos-py       · playback / volume)
      └─ HomeKit MCP    (homekit-py     · blinds / covers)
  → TTS local           (Piper · de_DE-thorsten-high)
  → Speaker
```

Internet is required only for the LLM in the MVP. Everything else runs on the local network.

---

## 1. Technology Decisions

### 1.1 Wake-Word — OpenWakeWord

Apache 2.0, fully local, custom model training pipeline available for "Pantau".

Ramp-up plan:
1. MVP placeholder: push-to-talk key or closest existing model ("hey jarvis")
2. Train a custom `pantau.tflite` via the OpenWakeWord training pipeline
3. Tune threshold with real room noise (start at 0.5)
4. Post-wake timeout: 4–8 s
5. Barge-in: v0.3

Porcupine is robust but requires a Picovoice AccessKey at init — a cloud/account dependency
that conflicts with the local-first architecture goal.

### 1.2 VAD — Silero VAD + faster-whisper filter

Two separate roles in the pipeline:

| Stage | Tool | Role |
|---|---|---|
| Pre-STT | Silero VAD | Speech onset detection; discards non-speech chunks before Whisper |
| Inside STT | faster-whisper `vad_filter=True` | Second pass; strips residual silence from the audio window |

Silero processes a 30 ms chunk in < 1 ms on a single CPU core — negligible overhead.

### 1.3 STT — faster-whisper (local, German)

`language="de"`, `beam_size=1`, `vad_filter=True`, `condition_on_previous_text=False`

`condition_on_previous_text=False` prevents hallucinated continuations from prior turns —
important for short command recognition.

| Hardware | Model | Runtime |
|---|---|---|
| Raspberry Pi / weak CPU | Vosk or `tiny` / `base` | whisper.cpp / Vosk |
| Mini PC CPU only | `small` int8 | faster-whisper |
| Modern CPU / 8 GB RAM | `medium` int8 | faster-whisper |
| NVIDIA GPU ≥ 4 GB | `small` or `medium` fp16 | faster-whisper |

Start with `small`. Upgrade only if German accuracy is insufficient.

### 1.4 LLM

**MVP (cloud):** `gpt-5.4-nano` — $0.20 / 1M input tokens, $1.25 / 1M output (May 2026).
Pure text routing for smart home commands costs fractions of a cent per call.
Do not use a realtime audio model if STT/TTS are local — they are priced separately and
significantly more expensive.

**Offline target:** Ollama + `qwen3` (or `qwen2.5:3b` for weaker hardware).
pydantic-ai supports it via `ollama:qwen3` with no other changes.

```python
# swap model string only — everything else stays identical
agent = Agent("ollama:qwen3", instructions=SYSTEM_PROMPT, toolsets=[home_mcp])
```

### 1.5 MCP Design — FastMCP Facade

A single `pantau-home-mcp` server exposes 7 opinionated, high-level tools.
The LLM sees exactly 7 tools instead of dozens of low-level device primitives —
fewer tools means faster and more reliable tool selection.

```
pantau_turn_on_room(room, brightness, color)
pantau_turn_off_room(room)
pantau_start_tv(activity)
pantau_power_off_tv()
pantau_set_blinds(room, position)
pantau_play_music(room, query)
pantau_set_volume(room, value_or_delta)
```

The facade also acts as a security perimeter: every device action goes through a named,
typed Python function with explicit validation. No raw device commands are ever exposed
to the LLM.

### 1.6 TTS — Piper TTS (local, German)

`de_DE-thorsten-high` voice. ONNX-based, < 300 ms synthesis on CPU, no internet required.

### 1.7 pydantic-deepagents — not used

pydantic-deepagents is built for autonomous planning agents (filesystem, subagents, shell).
Pantau is a reactive command executor: 1 utterance → 1 MCP tool → 1 response.
The framework would add planning overhead, filesystem tools in LLM context, and startup
latency with no benefit for this interaction model.

Reconsider only for v0.3 if multi-step scheduling commands are added
("Schalte abends um 18 Uhr alle Lichter ein und spiele Jazz").

---

## 2. Project Structure

```
pantau/
├── pyproject.toml
├── config/
│   ├── pantau.yaml              # base configuration
│   └── pantau.mcp.json          # MCP server definitions (reference)
├── pantau/
│   ├── __init__.py
│   ├── main.py                  # async event loop
│   ├── config.py                # PantauConfig (extends BaseConfig)
│   ├── audio/
│   │   ├── wakeword.py          # OpenWakeWord listener
│   │   ├── vad.py               # Silero VAD / endpointing
│   │   ├── stt.py               # faster-whisper transcription
│   │   └── tts.py               # Piper TTS synthesis + playback
│   ├── agent/
│   │   ├── runtime.py           # pydantic-ai Agent construction
│   │   ├── prompts.py           # SYSTEM_PROMPT
│   │   ├── fast_path.py         # deterministic pre-router
│   │   └── skills.py            # optional domain knowledge injection
│   └── home_mcp/
│       ├── server.py            # FastMCP facade (pantau-home-mcp)
│       ├── tools_harmony.py
│       ├── tools_hue.py
│       ├── tools_sonos.py
│       └── tools_homekit.py
└── evals/
    ├── commands_de.yaml
    └── test_agent_routing.py
```

**Python: 3.13+** — matches appkit-commons and the jenreh device library requirements.

---

## 3. Configuration

### 3.1 PantauConfig — `pantau/config.py`

Extends `BaseConfig` from `appkit-commons`. YAML is the base; environment variables
(prefix `PANTAU_`) override at runtime. Sensitive values use the `secret:` prefix and
are resolved automatically from environment variables or Azure Key Vault.

```python
from appkit_commons.configuration import BaseConfig, load_configuration, setup_logging


class SttConfig(BaseConfig):
    model_size: str = "small"   # tiny | base | small | medium | large-v3
    device: str = "auto"        # auto | cpu | cuda
    language: str = "de"


class TtsConfig(BaseConfig):
    model: str = "models/de_DE-thorsten-high.onnx"
    speak_rate: float = 1.0


class WakeWordConfig(BaseConfig):
    model: str = "models/pantau.tflite"
    threshold: float = 0.5
    post_wake_timeout_s: float = 6.0


class LlmConfig(BaseConfig):
    provider: str = "openai"           # openai | ollama
    model: str = "gpt-5.4-nano"
    api_key: str = "secret:openai_api_key"   # resolved from env OPENAI_API_KEY
    base_url: str | None = None        # set for Ollama: http://localhost:11434/v1


class HueConfig(BaseConfig):
    bridge_ip: str = "192.168.1.2"
    api_key: str = "secret:hue_api_key"


class HarmonyConfig(BaseConfig):
    host: str = "192.168.1.50"


class SonosConfig(BaseConfig):
    discovery_timeout_s: int = 5


class HomekitConfig(BaseConfig):
    pin: str = "secret:homekit_pin"
    allow_write_tools: bool = True


class PantauConfig(BaseConfig):
    model_config = {"env_prefix": "PANTAU_"}

    log_level: str = "INFO"

    wake_word: WakeWordConfig = WakeWordConfig()
    stt: SttConfig = SttConfig()
    tts: TtsConfig = TtsConfig()
    llm: LlmConfig = LlmConfig()
    hue: HueConfig = HueConfig()
    harmony: HarmonyConfig = HarmonyConfig()
    sonos: SonosConfig = SonosConfig()
    homekit: HomekitConfig = HomekitConfig()


def load_config(path: str = "config/pantau.yaml") -> PantauConfig:
    cfg = load_configuration(PantauConfig, path)
    setup_logging(level=cfg.log_level)
    return cfg
```

### 3.2 YAML base config — `config/pantau.yaml`

```yaml
log_level: INFO

wake_word:
  model: models/pantau.tflite
  threshold: 0.5
  post_wake_timeout_s: 6.0

stt:
  model_size: small
  device: auto
  language: de

tts:
  model: models/de_DE-thorsten-high.onnx
  speak_rate: 1.0

llm:
  provider: openai
  model: gpt-5.4-nano
  # api_key resolved via secret:openai_api_key → env var OPENAI_API_KEY

hue:
  bridge_ip: 192.168.1.2
  # api_key resolved via secret:hue_api_key

harmony:
  host: 192.168.1.50

sonos:
  discovery_timeout_s: 5

homekit:
  allow_write_tools: true
  # pin resolved via secret:homekit_pin
```

### 3.3 Environment variable overrides

appkit-commons uses `__` as the nested separator:

```bash
# override LLM for offline mode
export PANTAU_LLM__PROVIDER=ollama
export PANTAU_LLM__MODEL=qwen3
export PANTAU_LLM__BASE_URL=http://localhost:11434/v1

# override STT model
export PANTAU_STT__MODEL_SIZE=medium

# secrets (resolved via secret: prefix)
export OPENAI_API_KEY=sk-...
export HUE_API_KEY=abc123
export HOMEKIT_PIN=123-45-678
```

---

## 4. Core Implementation

### 4.1 Wake Word — `pantau/audio/wakeword.py`

```python
import asyncio
import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from pantau.config import WakeWordConfig


class WakeWordListener:
    SAMPLE_RATE = 16_000
    CHUNK_MS = 80       # 1280 samples — OpenWakeWord requirement

    def __init__(self, cfg: WakeWordConfig) -> None:
        self.threshold = cfg.threshold
        self.model = Model(
            wakeword_models=[cfg.model],
            inference_framework="tflite",
        )

    async def listen(self) -> None:
        """Blocks until wake word is detected."""
        loop = asyncio.get_event_loop()
        detected = asyncio.Event()

        def _callback(indata, frames, time, status):
            pcm = (indata[:, 0] * 32768).astype(np.int16)
            prediction = self.model.predict(pcm)
            if any(v >= self.threshold for v in prediction.values()):
                loop.call_soon_threadsafe(detected.set)

        chunk = int(self.SAMPLE_RATE * self.CHUNK_MS / 1000)
        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk,
            callback=_callback,
        ):
            await detected.wait()
```

### 4.2 VAD — `pantau/audio/vad.py`

```python
import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator


class SileroVAD:
    SAMPLE_RATE = 16_000

    def __init__(self) -> None:
        model = load_silero_vad()
        self.iterator = VADIterator(model, sampling_rate=self.SAMPLE_RATE)

    def is_speech(self, chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(chunk).float()
        result = self.iterator(tensor, return_seconds=False)
        return result is not None
```

### 4.3 STT — `pantau/audio/stt.py`

```python
import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pantau.config import SttConfig


class GermanSTT:
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION_S = 1.2
    MAX_DURATION_S = 10.0

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg
        compute = "float16" if cfg.device == "cuda" else "int8"
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=compute,
        )

    async def record_and_transcribe(self) -> str:
        audio = await asyncio.to_thread(self._record_until_silence)
        return await asyncio.to_thread(self._transcribe, audio)

    def _record_until_silence(self) -> np.ndarray:
        sr = 16_000
        frames: list[np.ndarray] = []
        silent_chunks = 0
        silence_limit = int(self.SILENCE_DURATION_S * sr / 512)
        max_chunks = int(self.MAX_DURATION_S * sr / 512)

        with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            blocksize=512) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(512)
                frames.append(chunk[:, 0])
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms < self.SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= silence_limit:
                        break
                else:
                    silent_chunks = 0
        return np.concatenate(frames)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
```

### 4.4 TTS — `pantau/audio/tts.py`

```python
import asyncio
import io
import sounddevice as sd
import soundfile as sf
from piper.voice import PiperVoice
from pantau.config import TtsConfig


class PiperTTS:
    def __init__(self, cfg: TtsConfig) -> None:
        self.voice = PiperVoice.load(cfg.model)

    async def speak(self, text: str) -> None:
        await asyncio.to_thread(self._synthesize_and_play, text)

    def _synthesize_and_play(self, text: str) -> None:
        buf = io.BytesIO()
        with sf.SoundFile(buf, mode="w", samplerate=22050,
                          channels=1, format="WAV") as f:
            for audio_bytes in self.voice.synthesize_stream_raw(text):
                f.buffer_write(audio_bytes, dtype="int16")
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        sd.play(data, sr, blocking=True)
```

### 4.5 Fast-Path Router — `pantau/agent/fast_path.py`

Handles the most frequent fixed phrases deterministically — no LLM call, no network round-trip.
Saves ~400–700 ms per matched command.

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FastPathResult:
    tool: str
    args: dict


_ROUTES: dict[str, FastPathResult] = {
    phrase: result
    for phrases, result in [
        (
            {"schalte den fernseher ein", "mach den fernseher an",
             "fernseher ein", "tv ein", "tv an"},
            FastPathResult("pantau_start_tv", {"activity": "Fernsehen"}),
        ),
        (
            {"schalte den fernseher aus", "fernseher aus", "tv aus"},
            FastPathResult("pantau_power_off_tv", {}),
        ),
        (
            {"schalte das wohnzimmer ein", "wohnzimmer ein",
             "licht im wohnzimmer an", "wohnzimmer licht an"},
            FastPathResult("pantau_turn_on_room", {"room": "Wohnzimmer"}),
        ),
        (
            {"schalte das wohnzimmer aus", "wohnzimmer aus",
             "licht im wohnzimmer aus"},
            FastPathResult("pantau_turn_off_room", {"room": "Wohnzimmer"}),
        ),
    ]
    for phrase in phrases
}


def fast_path(text: str) -> FastPathResult | None:
    return _ROUTES.get(text.lower().strip())
```

### 4.6 FastMCP Facade — `pantau/home_mcp/server.py`

```python
from __future__ import annotations
from fastmcp import FastMCP
from harmonyhub import HarmonyHubClient
from huehub import HueBridgeClient, load_config as load_hue_config
from sonospy import SonosClient
from homekit import HomeKitClient, load_config as load_homekit_config
from pantau.config import load_config

cfg = load_config()

mcp = FastMCP(
    name="pantau-home",
    instructions=(
        "Local smart-home tools for Pantau. "
        "Use for German voice commands. "
        "Never expose raw device IDs, tool names, or internal details."
    ),
)


@mcp.tool
async def pantau_start_tv(activity: str = "Fernsehen") -> str:
    """Startet eine Harmony-Aktivität. Standard: 'Fernsehen'."""
    async with HarmonyHubClient(cfg.harmony.host) as hub:
        await hub.start_activity(activity)
    return f"Aktivität '{activity}' gestartet."


@mcp.tool
async def pantau_power_off_tv() -> str:
    """Schaltet alle Harmony-Geräte aus."""
    async with HarmonyHubClient(cfg.harmony.host) as hub:
        await hub.turn_off()
    return "Alle Geräte ausgeschaltet."


@mcp.tool
async def pantau_turn_on_room(
    room: str,
    brightness: int | None = None,
    color: str | None = None,
) -> str:
    """Schaltet Hue-Lichter in einem Raum ein. room = Raumname auf Deutsch."""
    async with HueBridgeClient(load_hue_config()) as hue:
        await hue.turn_on_room(room, brightness=brightness, color=color or "warm")
    return f"{room} eingeschaltet."


@mcp.tool
async def pantau_turn_off_room(room: str) -> str:
    """Schaltet Hue-Lichter in einem Raum aus."""
    async with HueBridgeClient(load_hue_config()) as hue:
        await hue.set_room(room, on=False)
    return f"{room} ausgeschaltet."


@mcp.tool
async def pantau_set_blinds(room: str, position: int) -> str:
    """Steuert Rollos über HomeKit. position: 0 = geschlossen, 100 = geöffnet."""
    if not 0 <= position <= 100:
        raise ValueError("position muss zwischen 0 und 100 liegen")
    entity_id = f"cover.{room.lower().replace(' ', '_')}"
    async with HomeKitClient(load_homekit_config()) as hk:
        await hk.set_cover(entity_id, position=position)
    return f"Rollo in {room} auf {position} % gesetzt."


@mcp.tool
async def pantau_play_music(room: str, query: str) -> str:
    """Spielt Musik über Sonos in einem Raum."""
    sonos = SonosClient()
    speaker = await sonos.find_speaker(room)
    await speaker.play_uri(query)
    return f"Starte '{query}' in {room}."


@mcp.tool
async def pantau_set_volume(room: str, value_or_delta: int) -> str:
    """Setzt oder ändert Sonos-Lautstärke. Positiv = lauter, negativ = leiser."""
    sonos = SonosClient()
    speaker = await sonos.find_speaker(room)
    if 0 <= value_or_delta <= 100:
        await speaker.set_volume(value_or_delta)
    else:
        current = await speaker.get_volume()
        await speaker.set_volume(max(0, min(100, current + value_or_delta)))
    return f"Lautstärke in {room} angepasst."


if __name__ == "__main__":
    mcp.run()   # stdio by default
```

### 4.7 Agent — `pantau/agent/runtime.py`

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pantau.config import PantauConfig

SYSTEM_PROMPT = """
Du bist Pantau, ein lokaler Voice-Agent für Smart Home.

Aufgabe:
- Interpretiere deutsche Sprachbefehle.
- Rufe genau das passende Tool auf.
- Antworte kurz und natürlich auf Deutsch (maximal ein Satz).
- Nenne niemals Toolnamen, JSON, IDs oder interne Details.

Zuordnung:
- Fernseher, TV, Fernsehen, Apple TV, Receiver → pantau_start_tv / pantau_power_off_tv
- Licht, Lampe, Szene, Raum → pantau_turn_on_room / pantau_turn_off_room
- Musik, Radio, Lautstärke, Pause, Weiter → pantau_play_music / pantau_set_volume
- Rollo, Jalousie, Fensterblende, hoch, runter → pantau_set_blinds

Sicherheitsregeln:
- Keine Websuche, keine Aktionen außerhalb Smart Home.
- Bei unklarem Raum oder Zielgerät einmal kurz nachfragen.
- Bei gefährlichen Aktionen nicht raten.
"""


def build_agent(cfg: PantauConfig) -> Agent:
    # resolve model string
    if cfg.llm.provider == "ollama":
        model_str = f"ollama:{cfg.llm.model}"
    else:
        model_str = f"openai:{cfg.llm.model}"

    home_mcp = MCPServerStdio(
        "python",
        args=["-m", "pantau.home_mcp.server"],
        timeout=10,
    )
    return Agent(
        model_str,
        instructions=SYSTEM_PROMPT,
        toolsets=[home_mcp],
    )
```

### 4.8 Main Loop — `pantau/main.py`

```python
import asyncio
import logging
from pantau.config import load_config
from pantau.audio.wakeword import WakeWordListener
from pantau.audio.stt import GermanSTT
from pantau.audio.tts import PiperTTS
from pantau.agent.runtime import build_agent
from pantau.agent.fast_path import fast_path

logger = logging.getLogger("pantau")


async def main() -> None:
    cfg      = load_config("config/pantau.yaml")   # also calls setup_logging()
    wakeword = WakeWordListener(cfg.wake_word)
    stt      = GermanSTT(cfg.stt)
    tts      = PiperTTS(cfg.tts)
    agent    = build_agent(cfg)

    await tts.speak("Pantau ist bereit.")
    logger.info("Warte auf Aktivierungswort …")

    while True:
        try:
            await wakeword.listen()
            await tts.speak("Ja?")

            text = await stt.record_and_transcribe()
            if not text:
                continue
            logger.info("Erkannt: %s", text)

            # fast path: deterministic, no LLM
            match = fast_path(text)
            if match:
                logger.info("Fast-path: %s %s", match.tool, match.args)
                response = await call_mcp_tool(match.tool, match.args)
            else:
                async with agent.run_mcp_servers():
                    result = await agent.run(text)
                response = result.output

            logger.info("Antwort: %s", response)
            await tts.speak(response)

        except Exception:
            logger.exception("Unbehandelter Fehler")
            await tts.speak("Es gab einen Fehler. Bitte versuche es erneut.")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Skills (optionale Domänenwissen-Injektion)

Skills sind Markdown-Dateien, die beim Start als zusätzliche System-Prompt-Sektionen
geladen werden — kein Extra-Framework nötig.

```python
# pantau/agent/skills.py
from pathlib import Path


def load_skills(skills_dir: Path = Path("pantau/skills")) -> str:
    sections = [f.read_text() for f in sorted(skills_dir.glob("*.md"))]
    return "\n\n---\n\n".join(sections)
```

```markdown
<!-- pantau/skills/hue_rooms.md -->
## Hue Raumzuordnung

| Sprachbefehl          | Hue Gruppenname |
|-----------------------|-----------------|
| Wohnzimmer            | Living Room     |
| Schlafzimmer          | Bedroom         |
| Küche                 | Kitchen         |
| Bad / Badezimmer      | Bathroom        |
| Büro / Arbeitszimmer  | Office          |
| Überall / Alle Räume  | (all groups)    |
```

```markdown
<!-- pantau/skills/harmony_activities.md -->
## Harmony Aktivitäten

| Sprachbefehl              | Harmony Aktivität |
|---------------------------|-------------------|
| Fernseher / TV            | Fernsehen         |
| Apple TV                  | AppleTV           |
| Spielen / PlayStation     | PlayStation       |
| Alles aus / Ausschalten   | Alles aus         |
```

---

## 6. Evals — `evals/`

```yaml
# evals/commands_de.yaml
- input: "schalte den Fernseher ein"
  expected_tool: pantau_start_tv
  expected_args: {activity: Fernsehen}

- input: "mach das Wohnzimmer an"
  expected_tool: pantau_turn_on_room
  expected_args: {room: Wohnzimmer}

- input: "Rollo im Schlafzimmer auf 30 Prozent"
  expected_tool: pantau_set_blinds
  expected_args: {room: Schlafzimmer, position: 30}

- input: "mach Musik im Büro leiser"
  expected_tool: pantau_set_volume
  expected_args: {room: Büro}

- input: "schalte alle Lichter aus"
  expected_tool: pantau_turn_off_room
```

Run before and after every LLM swap (cloud → local) to quantify routing accuracy.

---

## 7. Dependencies — `pyproject.toml`

```toml
[project]
name = "pantau"
requires-python = ">=3.13"

dependencies = [
    # AppKit infrastructure
    "appkit-commons>=0.1.0",

    # Agent
    "pydantic-ai-slim[openai]>=0.4.0",
    "fastmcp>=2.0.0",

    # Voice pipeline
    "faster-whisper>=1.0.0",
    "openwakeword>=0.6.0",
    "silero-vad>=5.0.0",
    "piper-tts>=1.2.0",
    "sounddevice>=0.4.6",
    "soundfile>=0.12.1",
    "numpy>=2.0",
    "torch>=2.3",

    # Smart home (jenreh repos)
    "harmonyhub-py @ git+https://github.com/jenreh/harmonyhub-py",
    "huehub-py @ git+https://github.com/jenreh/huehub-py",
    "sonos-py @ git+https://github.com/jenreh/sonos-py",
    "homekit-py @ git+https://github.com/jenreh/homekit-py",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

> `pydantic-settings` and `pyyaml` are **not** listed directly — they are transitive
> dependencies pulled in by `appkit-commons`.

---

## 8. Latency Budget

| Stage | Target |
|---|---|
| Wake word detection | < 100 ms |
| Activation chime ("Ja?") | < 200 ms |
| VAD + audio capture | 1–4 s |
| STT (`small`, CPU) | ~1 s per 3 s audio |
| Fast-path check | < 1 ms |
| LLM round-trip (gpt-5.4-nano) | 300–700 ms |
| MCP tool execution (LAN) | 50–300 ms |
| TTS synthesis + playback | 200–400 ms |
| **Total — fast path** | **~2–4 s** |
| **Total — LLM path** | **~2.5–5 s** |

---

## 9. Network Constraints

```
Allowed:
  127.0.0.1             MCP stdio / localhost
  LAN IPs               Hue bridge, Harmony Hub, Sonos, HomeKit
  cloud LLM endpoint    MVP only (one domain allowlist entry)

Blocked:
  Cloud STT / TTS
  Web search / browser tools
  Public MCP servers
  Shell / filesystem tools
```

---

## 10. Implementation Phases

**Phase 1 — Text-only agent**
Wire pydantic-ai to the FastMCP facade. Test all commands as plain text.
Run the eval set. Verify tool calls before adding any voice components.

**Phase 2 — Voice MVP**
Add wake word, VAD, STT, TTS. Keep cloud LLM.
Tune OpenWakeWord threshold against real room noise.

**Phase 3 — Local LLM**
Run Ollama + qwen3. Compare eval accuracy vs gpt-5.4-nano.
Extend fast-path routes to compensate for any accuracy loss.

**Phase 4 — Production hardening**
Audit log for every tool call · room/device alias expansion via skills ·
confirmation prompts for risky actions · per-stage latency metrics ·
fallback responses for unreachable devices · barge-in support.
