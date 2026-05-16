# Pantau — Voice Agent: Implementation Concept v2

*Merged from two independent reviews*

---

## 1. Executive Summary

Pantau is a local-first, German-language voice agent for smart home control. It chains **wake-word → VAD → STT → fast-path router → agent → MCP facade → TTS**, requiring internet only for the LLM (MVP). All device control runs over the local network with no cloud dependency.

---

## 2. Technology Decisions

### 2.1 Wake-Word Detection

**→ OpenWakeWord** (Apache 2.0)

Runs fully local on CPU. A custom model for "Pantau" is needed since it is not a built-in wake word. Practical ramp-up:

1. MVP placeholder: use a push-to-talk key, or the closest existing model ("hey jarvis")
2. Train a custom model using OpenWakeWord's training pipeline
3. Tune threshold against real room noise
4. Set a post-wake timeout of 4–8 s
5. "Barge-in" (interrupting agent speech): defer to v0.3

**Porcupine** is viable and robust on-device, but requires a Picovoice AccessKey at init — a cloud dependency that conflicts with the no-account architecture goal.

### 2.2 VAD / Endpointing

**→ Silero VAD** as a dedicated pipeline stage, *and* `vad_filter=True` inside faster-whisper

Two distinct roles:

| Stage | Tool | Purpose |
|---|---|---|
| Pre-STT | Silero VAD | Detect speech onset; discard non-speech chunks before sending to Whisper; acts as endpointing |
| Inside STT | faster-whisper `vad_filter` | Second pass; strips remaining silence from the audio window before decoding |

Silero VAD processes a 30 ms chunk in < 1 ms on a single CPU core — negligible overhead for the latency budget.

### 2.3 Speech-to-Text (local, German)

**→ faster-whisper** with `language="de"`, `beam_size=1`, `condition_on_previous_text=False`

`condition_on_previous_text=False` is important for command recognition: it prevents Whisper from hallucinating continuations of prior transcriptions across turns.

| Hardware | Model | Runtime | Notes |
|---|---|---|---|
| Raspberry Pi / weak CPU | Vosk or whisper.cpp tiny/base | Vosk / whisper.cpp | Lower accuracy, lower latency |
| Mini PC CPU only | `small` int8 | faster-whisper | Good MVP baseline |
| Modern CPU / 8 GB RAM | `medium` int8 | faster-whisper | Better German WER |
| NVIDIA GPU ≥ 4 GB | `small` or `medium` fp16 | faster-whisper | < 0.5 s latency |
| Quality-first | `large-v3` or `turbo` | faster-whisper / whisper.cpp | Likely overkill for commands |

Start with `small`. Upgrade only if German command accuracy is insufficient on your hardware.

### 2.4 LLM

#### MVP — Cloud, fast and cheap

**→ gpt-5.4-nano** (`$0.20 / 1M` input, `$1.25 / 1M` output — May 2026 pricing)

Pure text routing for smart home commands costs fractions of a cent per command. Do not use a realtime audio model for MVP if STT/TTS are local — they are priced separately and are significantly more expensive.

**gpt-5.4-mini** is stronger but ~4× more expensive; useful if tool-selection accuracy on the eval set is insufficient.

**Gemini 2.0 Flash** is a valid alternative with comparable cost and excellent German support.

#### Offline target

**→ Ollama + qwen3** (or qwen2.5:3b for weaker hardware)

pydantic-ai supports Ollama via `ollama:qwen3` out of the box. llama.cpp is an alternative; prefer models with native tool-call format handlers (Llama 3.x, Qwen 2.5, Hermes, Mistral Nemo).

```python
# Swap the model string; everything else stays identical
agent = Agent("ollama:qwen3", instructions=SYSTEM_PROMPT, toolsets=[home])
```

### 2.5 MCP Design: Two Options

#### Option A — Direct server wiring (fastest to implement)

Wire all four device MCP servers directly into pydantic-ai:

```json
// pantau.mcp.json
{
  "mcpServers": {
    "harmony": {
      "command": "harmony-mcp",
      "env": { "HARMONY_HUB_HOST": "${HARMONY_HUB_HOST}" }
    },
    "hue":     { "command": "hue-mcp" },
    "sonos":   { "command": "sonos-mcp" },
    "homekit": {
      "command": "homekit-mcp",
      "env": { "HOMEKIT_MCP__ALLOW_WRITE_TOOLS": "true" }
    }
  }
}
```

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_servers

servers = load_mcp_servers("pantau.mcp.json")
agent = Agent("openai:gpt-5.4-nano", instructions=SYSTEM_PROMPT, toolsets=servers)
```

**Problem**: The LLM sees many low-level device tools. This increases latency, token consumption, and the probability of wrong tool selection.

#### Option B — FastMCP facade (recommended)

A single `pantau-home-mcp` server exposes a small, opinionated set of high-level tools. The LLM sees exactly 7 tools instead of dozens of low-level primitives.

```python
pantau_turn_on_room(room, brightness, color)
pantau_turn_off_room(room)
pantau_start_tv(activity)
pantau_power_off_tv()
pantau_set_blinds(room, position)
pantau_play_music(room, query)
pantau_set_volume(room, value_or_delta)
```

**Why Option B is better for voice:**

- Fewer tools = faster, more reliable tool selection
- Semantic tool names match natural language intent
- Device-specific quirks are hidden from the LLM
- Easier to add room/device aliases and safety checks in one place

### 2.6 Text-to-Speech (local, German)

**→ Piper TTS** with `de_DE-thorsten-high` voice

ONNX-based, extremely fast (< 300 ms for typical agent responses on CPU), no internet required. Kokoro-82M is higher quality but needs more RAM/VRAM; useful for v0.3.

### 2.7 Should pydantic-deepagents be used?

**For MVP: No. For a skills-heavy v0.3: maybe.**

pydantic-deepagents is built for autonomous planning agents (filesystem access, subagent delegation, multi-step task execution). Pantau is a reactive command executor: 1 command → 1 MCP tool → 1 response. Adding pydantic-deepagents would introduce:

- Planning and filesystem tools in the LLM context (security risk for home control)
- Higher startup latency per request
- Architectural complexity with no benefit for this interaction model

Consider it **only** if you later want SKILL.md auto-discovery and injection, persistent memory across sessions, lifecycle hooks for audit logging, or a more autonomous multi-step agent (e.g. "Schalte abends um 18 Uhr alle Lichter ein und spiele Jazz"). That is a v0.3 concern.

---

## 3. Project Structure

```bash
pantau/
├── pyproject.toml
├── config/
│   ├── pantau.toml              # device IPs, LLM settings, wake word config
│   └── pantau.mcp.json          # MCP server definitions (Option A reference)
├── pantau/
│   ├── __init__.py
│   ├── main.py                  # async event loop
│   ├── config.py                # pydantic-settings config model
│   ├── audio/
│   │   ├── microphone.py        # audio capture (sounddevice)
│   │   ├── wakeword.py          # OpenWakeWord listener
│   │   ├── vad.py               # Silero VAD / endpointing
│   │   ├── stt.py               # faster-whisper transcription
│   │   └── tts.py               # Piper TTS synthesis + playback
│   ├── agent/
│   │   ├── runtime.py           # pydantic-ai Agent construction
│   │   ├── prompts.py           # SYSTEM_PROMPT, routing prompt
│   │   ├── fast_path.py         # deterministic pre-router
│   │   └── mcp_config.py        # MCP server wiring (Option B)
│   └── home_mcp/
│       ├── server.py            # FastMCP facade (pantau-home-mcp)
│       ├── tools_harmony.py
│       ├── tools_hue.py
│       ├── tools_sonos.py
│       └── tools_homekit.py
└── evals/
    ├── commands_de.yaml         # German command test cases
    └── test_agent_routing.py    # pytest eval harness
```

**Python version: 3.14+** — required to match the `jenreh/*` device library dependencies.

---

## 4. Core Implementation

### 4.1 Wake Word — `pantau/audio/wakeword.py`

```python
import asyncio
import numpy as np
import sounddevice as sd
from openwakeword.model import Model


class WakeWordListener:
    SAMPLE_RATE = 16_000
    CHUNK_MS = 80  # 1280 samples — OpenWakeWord requirement
    THRESHOLD = 0.5

    def __init__(self, model_path: str = "hey_jarvis"):
        self.model = Model(wakeword_models=[model_path], inference_framework="tflite")

    async def listen(self) -> None:
        """Blocks until wake word is detected."""
        loop = asyncio.get_event_loop()
        detected = asyncio.Event()

        def audio_callback(indata, frames, time, status):
            pcm = (indata[:, 0] * 32768).astype(np.int16)
            prediction = self.model.predict(pcm)
            if any(v >= self.THRESHOLD for v in prediction.values()):
                loop.call_soon_threadsafe(detected.set)

        chunk = int(self.SAMPLE_RATE * self.CHUNK_MS / 1000)
        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk,
            callback=audio_callback,
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
    CHUNK_SIZE = 512  # ~32 ms

    def __init__(self):
        self.model = load_silero_vad()
        self.iterator = VADIterator(self.model, sampling_rate=self.SAMPLE_RATE)

    def is_speech(self, chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(chunk).float()
        result = self.iterator(tensor, return_seconds=False)
        return result is not None  # dict returned only during speech
```

### 4.3 STT — `pantau/audio/stt.py`

```python
import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class GermanSTT:
    SAMPLE_RATE = 16_000
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION_S = 1.2
    MAX_DURATION_S = 10.0

    def __init__(self, model_size: str = "small", device: str = "auto"):
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute)

    def transcribe_command(self, wav_path: str) -> str:
        segments, _ = self.model.transcribe(
            wav_path,
            language="de",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,  # prevents hallucination carry-over
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def record_and_transcribe(self) -> str:
        audio = await asyncio.to_thread(self._record_until_silence)
        return await asyncio.to_thread(self._transcribe_array, audio)

    def _record_until_silence(self) -> np.ndarray:
        frames = []
        silent_chunks = 0
        silence_limit = int(self.SILENCE_DURATION_S * self.SAMPLE_RATE / 512)
        max_chunks = int(self.MAX_DURATION_S * self.SAMPLE_RATE / 512)
        with sd.InputStream(
            samplerate=self.SAMPLE_RATE, channels=1, dtype="float32", blocksize=512
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(512)
                frames.append(chunk[:, 0])
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms < self.SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= silence_limit:
                        break
                else:
                    silent_chunks = 0
        return np.concatenate(frames)

    def _transcribe_array(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="de",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
```

### 4.4 Fast-Path Pre-Router — `pantau/agent/fast_path.py`

Handles the most common fixed phrases deterministically, bypassing the LLM entirely. This eliminates network round-trip latency (~400–700 ms) for high-frequency commands.

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FastPathResult:
    tool: str
    args: dict


_FAST_PATHS: dict[frozenset[str], FastPathResult] = {
    frozenset({
        "schalte den fernseher ein",
        "mach den fernseher an",
        "fernseher ein",
        "tv ein",
        "tv an",
        "fernseher an",
    }): FastPathResult("pantau_start_tv", {"activity": "Fernsehen"}),
    frozenset({
        "schalte den fernseher aus",
        "fernseher aus",
        "tv aus",
        "mach den fernseher aus",
    }): FastPathResult("pantau_power_off_tv", {}),
    frozenset({
        "schalte das wohnzimmer ein",
        "wohnzimmer ein",
        "licht im wohnzimmer an",
        "wohnzimmer licht an",
    }): FastPathResult("pantau_turn_on_room", {"room": "Wohnzimmer"}),
    frozenset({
        "schalte das wohnzimmer aus",
        "wohnzimmer aus",
        "licht im wohnzimmer aus",
        "wohnzimmer licht aus",
    }): FastPathResult("pantau_turn_off_room", {"room": "Wohnzimmer"}),
}
# Build a flat lookup from normalized phrase → result
_LOOKUP: dict[str, FastPathResult] = {
    phrase: result for phrases, result in _FAST_PATHS.items() for phrase in phrases
}


def fast_path(text: str) -> FastPathResult | None:
    return _LOOKUP.get(text.lower().strip())
```

The main loop checks fast_path first. Only on a miss does it invoke the agent. This also works as a latency benchmark: if the agent route takes > 2 s for a phrase that has a fast path, add it.

### 4.5 FastMCP Facade — `pantau/home_mcp/server.py`

```python
from __future__ import annotations
import os
from fastmcp import FastMCP
from harmonyhub import HarmonyHubClient
from huehub import HueBridgeClient, load_config as load_hue_config
from sonospy import SonosClient
from homekit import HomeKitClient, load_config as load_homekit_config

mcp = FastMCP(
    name="pantau-home",
    instructions=(
        "Local smart-home tools for Pantau. "
        "Use these for German voice commands. "
        "Never expose raw device IDs or internal tool names."
    ),
)


@mcp.tool
async def pantau_start_tv(activity: str = "Fernsehen") -> str:
    """Startet eine Harmony-Aktivität. Standardmäßig 'Fernsehen'."""
    async with HarmonyHubClient(os.environ["HARMONY_HUB_HOST"]) as hub:
        await hub.start_activity(activity)
    return f"Aktivität '{activity}' gestartet."


@mcp.tool
async def pantau_power_off_tv() -> str:
    """Schaltet alle Harmony-Geräte aus."""
    async with HarmonyHubClient(os.environ["HARMONY_HUB_HOST"]) as hub:
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
    """Steuert Rollos/Jalousien über HomeKit. position: 0 = geschlossen, 100 = offen."""
    if not 0 <= position <= 100:
        raise ValueError("position muss zwischen 0 und 100 liegen")
    entity_id = f"cover.{room.lower().replace(' ', '_')}"
    async with HomeKitClient(load_homekit_config()) as hk:
        await hk.set_cover(entity_id, position=position)
    return f"Rollo in {room} auf {position} % gesetzt."


@mcp.tool
async def pantau_play_music(room: str, query: str) -> str:
    """Spielt Musik über Sonos in einem Raum ab."""
    sonos = SonosClient()
    speaker = await sonos.find_speaker(room)
    await speaker.play_uri(query)
    return f"Starte '{query}' in {room}."


@mcp.tool
async def pantau_set_volume(room: str, value_or_delta: int) -> str:
    """Setzt oder ändert die Sonos-Lautstärke. Positive = lauter, negative = leiser."""
    sonos = SonosClient()
    speaker = await sonos.find_speaker(room)
    if value_or_delta > 0 and value_or_delta <= 100:
        await speaker.set_volume(value_or_delta)
    else:
        current = await speaker.get_volume()
        await speaker.set_volume(max(0, min(100, current + value_or_delta)))
    return f"Lautstärke in {room} angepasst."


if __name__ == "__main__":
    mcp.run()  # stdio by default
```

**HomeKit write tool safety note:** `homekit-py` disables MCP write tools by default. The facade already scopes access — only `pantau_set_blinds` calls HomeKit write operations, and only via the typed facade, never via raw HomeKit MCP exposure.

### 4.6 Agent — `pantau/agent/runtime.py`

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

SYSTEM_PROMPT = """
Du bist Pantau, ein lokaler Voice-Agent für Smart Home.

Aufgabe:
- Interpretiere deutsche Sprachbefehle.
- Rufe genau das passende Tool auf.
- Antworte kurz und natürlich auf Deutsch (maximal ein Satz).
- Nenne niemals Toolnamen, JSON, IDs oder interne Details.

Zuordnung:
- Fernseher, TV, Fernsehen, Apple TV, Receiver → Harmony (pantau_start_tv / pantau_power_off_tv)
- Licht, Lampe, Szene, Raum → Hue (pantau_turn_on_room / pantau_turn_off_room)
- Musik, Radio, Lautstärke, Pause, Weiter → Sonos (pantau_play_music / pantau_set_volume)
- Rollo, Jalousie, Fensterblende, hoch, runter → HomeKit (pantau_set_blinds)

Sicherheitsregeln:
- Keine Websuche, keine Aktionen außerhalb Smart Home.
- Bei unklarem Raum oder Zielgerät einmal kurz nachfragen.
- Bei gefährlichen Aktionen nicht raten.
"""


def build_agent() -> Agent:
    home_mcp = MCPServerStdio(
        "python",
        args=["-m", "pantau.home_mcp.server"],
        timeout=10,
    )
    return Agent(
        "openai:gpt-5.4-nano",
        instructions=SYSTEM_PROMPT,
        toolsets=[home_mcp],
    )
```

### 4.7 Main Loop — `pantau/main.py`

```python
import asyncio
import logging
from pantau.audio.wakeword import WakeWordListener
from pantau.audio.stt import GermanSTT
from pantau.audio.tts import PiperTTS
from pantau.agent.runtime import build_agent
from pantau.agent.fast_path import fast_path

logger = logging.getLogger("pantau")


async def main() -> None:
    logger.info("Pantau startet …")
    wakeword = WakeWordListener(model_path="hey_jarvis")  # replace with pantau.tflite
    stt = GermanSTT(model_size="small", device="auto")
    tts = PiperTTS(model_path="de_DE-thorsten-high.onnx")
    agent = build_agent()

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

            # --- fast path: no LLM call ---
            match = fast_path(text)
            if match:
                logger.info("Fast-path: %s %s", match.tool, match.args)
                # call the MCP server directly (or via a thin local client)
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
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

### 4.8 TTS — `pantau/audio/tts.py`

```python
import asyncio
import io
import sounddevice as sd
import soundfile as sf
from piper.voice import PiperVoice


class PiperTTS:
    def __init__(self, model_path: str = "de_DE-thorsten-high.onnx"):
        self.voice = PiperVoice.load(model_path)

    async def speak(self, text: str) -> None:
        await asyncio.to_thread(self._synthesize_and_play, text)

    def _synthesize_and_play(self, text: str) -> None:
        buf = io.BytesIO()
        with sf.SoundFile(
            buf, mode="w", samplerate=22050, channels=1, format="WAV"
        ) as f:
            for audio_bytes in self.voice.synthesize_stream_raw(text):
                f.buffer_write(audio_bytes, dtype="int16")
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        sd.play(data, sr, blocking=True)
```

---

## 5. Agent Routing Prompt (Tuned)

```markdown
Du bist Pantau, ein lokaler Voice-Agent für Smart Home.

Aufgabe:
- Interpretiere deutsche Sprachbefehle.
- Rufe genau das passende Tool auf.
- Antworte kurz und natürlich auf Deutsch.
- Nenne niemals Toolnamen, JSON, IDs oder interne Details.

Zuordnung:
- Fernseher, TV, Fernsehen, Apple TV, Receiver → Harmony.
- Licht, Lampe, Szene, Raumbeleuchtung → Hue.
- Musik, Radio, Lautstärke, Pause, Weiter → Sonos.
- Rollo, Jalousie, Fensterblende, hoch, runter, Prozent → HomeKit.

Sicherheitsregeln:
- Keine Websuche.
- Keine Aktionen außerhalb Smart Home.
- Bei unklaren Räumen oder Zielgeräten einmal kurz nachfragen.
- Bei gefährlichen Aktionen nicht raten.
```

---

## 6. Skills (Optional Domain Knowledge Injection)

Skills are plain markdown files loaded and appended to the system prompt at startup — no extra framework required.

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

| Sprachbefehl              | Hue Gruppenname |
|---------------------------|-----------------|
| Wohnzimmer                | Living Room     |
| Schlafzimmer              | Bedroom         |
| Küche                     | Kitchen         |
| Bad / Badezimmer          | Bathroom        |
| Büro / Arbeitszimmer      | Office          |
| Überall / Alle Räume      | (all groups)    |
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

## 7. Evals — `evals/`

**This layer is missing from most voice agent MVPs and should not be skipped.**

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
  expected_args: {room: Büro}   # value_or_delta can vary

- input: "schalte alle Lichter aus"
  expected_tool: pantau_turn_off_room
  expected_args: {room: Überall}
```

```python
# evals/test_agent_routing.py
import pytest
import yaml
from pantau.agent.runtime import build_agent

CASES = yaml.safe_load(open("evals/commands_de.yaml"))


@pytest.mark.parametrize("case", CASES)
async def test_routing(case):
    agent = build_agent()
    async with agent.run_mcp_servers():
        result = await agent.run(case["input"])
    # inspect result.tool_calls for correct tool + args
    ...
```

Run evals before and after every LLM swap (cloud → local) to quantify routing accuracy loss.

---

## 8. Security / Network Constraints

For the MVP, apply network constraints at the process level or via firewall:

```markdown
Allowed:
  127.0.0.1          (MCP stdio / localhost HTTP)
  LAN IPs            (Hue bridge, Harmony Hub, Sonos, HomeKit accessories)
  cloud LLM endpoint (MVP only — one domain allowlist entry)

Blocked:
  Cloud STT
  Cloud TTS
  Web search / browser tools
  Public MCP servers
  Arbitrary shell execution
```

The pantau-ai agent must never be given shell, filesystem, or browser tools. The FastMCP facade acts as a safety perimeter: every action goes through a named, typed Python function with explicit validation.

---

## 9. Dependencies — `pyproject.toml`

```toml
[project]
name = "pantau"
requires-python = ">=3.14"

dependencies = [
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
    "torch>=2.3",          # for Silero VAD

    # Smart home (jenreh repos)
    "harmonyhub-py @ git+https://github.com/jenreh/harmonyhub-py",
    "huehub-py @ git+https://github.com/jenreh/huehub-py",
    "sonos-py @ git+https://github.com/jenreh/sonos-py",
    "homekit-py @ git+https://github.com/jenreh/homekit-py",

    # Utilities
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

---

## 10. Latency Budget

| Stage | Target |
| --- | --- |
| Wake word detection | < 100 ms |
| Activation chime ("Ja?") | < 200 ms |
| VAD + audio capture | 1–4 s (depends on command length) |
| STT (`small`, CPU) | ~ 1 s per 3 s audio |
| Fast-path check | < 1 ms |
| LLM (gpt-5.4-nano, cloud) | 300–700 ms |
| MCP tool execution | 50–300 ms (LAN) |
| TTS synthesis + play | 200–400 ms |
| **Total (fast path)** | **~2–4 s** |
| **Total (LLM path)** | **~2.5–5 s** |

---

## 11. Implementation Phases

### Phase 1 — Text-only agent (validate routing)

Wire pydantic-ai to the FastMCP facade. Test commands as plain text. Run the eval set. Verify tool calls before adding any voice components.

### Phase 2 — Voice MVP

Add wake word, VAD, STT, TTS. Keep cloud LLM. Tune thresholds against real room noise.

### Phase 3 — Local LLM

Run Ollama + qwen3. Compare tool-selection accuracy against the eval set vs gpt-5.4-nano. Adjust the fast-path to compensate for any accuracy loss. Keep the FastMCP facade small — smaller tool surface = more reliable local model routing.

### Phase 4 — Production hardening

- Audit log for every tool call (tool name, args, response, latency)
- Room/device alias expansion (skills files)
- Confirmation prompts for risky actions (e.g. "Alle Geräte ausschalten?")
- Latency metrics per stage (wake / STT / LLM / tool / TTS)
- Fallback responses when a device is unreachable
- "Barge-in" (interrupting agent TTS with a new wake word)

---

## 12. Comparison: What Each Review Contributed

| Topic | v1 (first review) | v2 addition (second review) |
| --- | --- | --- |
| Architecture diagram | ASCII art | ✔ (this document) |
| Wake word | OpenWakeWord + async code | Same; added Porcupine tradeoff |
| VAD | Inside faster-whisper only | **Silero VAD as own stage** |
| STT code | Full async capture + transcribe | `condition_on_previous_text=False` |
| MCP design | Option B only | **Option A vs B comparison** |
| Fast-path router | ❌ | **Added — major latency win** |
| Evals layer | ❌ | **Added — eval YAML + pytest** |
| Project structure | Flat modules | **audio/ agent/ home_mcp/ evals/** |
| LLM recommendation | Gemini 2.0 Flash | **gpt-5.4-nano with exact pricing** |
| Python version | Not specified | **3.14+ (matches jenreh deps)** |
| Network constraints | Config only | **Explicit allowed/blocked list** |
| HomeKit write safety | Not mentioned | **Explicit safety note** |
| Implementation phases | Simple roadmap | **4 structured phases** |
| pydantic-deepagents | Skip for MVP | Same conclusion, expanded reasoning |
