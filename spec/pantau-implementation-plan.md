# Pantau Implementation Task Prompt

## *Local German Voice Agent — All Phases*

---

## 🎯 Goal

**Complete a fully functional local German voice agent** that:

- Wakes on "Pantau" voice command
- Transcribes German speech to text in real-time
- Routes commands via a deterministic fast-path intent/entity recognizer or LLM
- Executes smart-home actions (lights, TV, speakers, blinds) through MCP tools
- Responds naturally in German via local TTS

**Success = end-to-end voice loop working** (wake → listen → respond → play audio) **within latency targets** (2–4s fast-path, 2.5–5s LLM), **with all tests passing** and **no cloud dependencies except MVP LLM** (swap to local Ollama later).

---

## Executive Summary

**What:** Implement a local German voice agent that listens for a wake word, transcribes speech to text, routes commands through a deterministic fast-path DSL recognizer or an LLM, and executes smart-home device actions through a FastMCP facade.

**Scope:** All 4 implementation phases — from text-only agent validation to production-hardened voice system.

**Audience:** Implementation engineer (assumes Python async, pydantic-ai, MCP, audio I/O familiarity).

**Current State:** ~50% complete. Working: seed fast-path router, MCP facade, device knowledge base. Missing: fast-path DSL redesign, audio pipeline, orchestration, config API alignment, pydantic-ai API update.

**Estimated Effort:** 8–12 hours (Phases 1–3 core); 3–5 hours optional Phase 4.

---

## Current Project Status

### ✅ Working Components

- **Fast-path router** (`pantau/agent/fast_path.py`) — working seed implementation; currently a phrase dictionary that should be replaced by a DSL-based recognizer
- **MCP facade** (`pantau/home_mcp/server.py`) — 7 high-level device tools (Harmony, Hue, Sonos, HomeKit)
- **Device knowledge base** (`pantau/skills/`) — Hue room names, Harmony activities
- **Config system** (`pantau/config.py`) — Full PantauConfig hierarchy (needs refactor)
- **Test suite** — fast_path, skills, MCP composition tests passing

### 🔴 Critical Gaps

1. **Fast-path design mismatch** — current implementation is a flat phrase dictionary; target design is an intent/entity/command DSL recognizer
2. **Audio stack missing** — no wake-word detection, VAD, STT, or TTS
3. **No orchestration** — no async event loop; no `main.py` entry point
4. **Config API mismatch** — current code uses `ApplicationConfig` + `configure()`; tests expect `PantauConfig` + `load_config()`
5. **Pydantic-ai deprecated** — `runtime.py` uses old `MCPServerStdio` API
6. **No way to run** — missing main entry point

### Files to Modify / Create

| File | Action | Priority |
| --- | --- | --- |
| `pantau/config.py` | Refactor | HIGH — blocks Phase 1 |
| `pantau/agent/fast_path.py` | Redesign | HIGH — replace phrase dict with DSL recognizer |
| `tests/agent/test_fast_path.py` | Rewrite | HIGH — validate DSL matching and command resolution |
| `pantau/agent/prompts.py` | Align | MEDIUM — remove stale fast-path command assumptions |
| `configuration/fast_path_rules.yaml` | Create | MEDIUM — declarative DSL rules file if externalized |
| `pantau/agent/runtime.py` | Rewrite | HIGH — fix pydantic-ai API |
| `pantau/audio/wakeword.py` | Create | HIGH — Phase 2 |
| `pantau/audio/vad.py` | Create | HIGH — Phase 2 |
| `pantau/audio/stt.py` | Create | HIGH — Phase 2 |
| `pantau/audio/tts.py` | Create | HIGH — Phase 2 |
| `pantau/main.py` | Create | HIGH — Phase 1 |
| `pyproject.toml` | Update | HIGH — add audio deps |
| `tests/test_config.py` | Update | HIGH — align to new API |

---

## Implementation Phases

### **Phase 1: Foundation — Text-Only Agent (2–3 hours)**

**Goal:** Validate text-only agent; replace the seed fast-path phrase dictionary with a deterministic DSL recognizer; wire pydantic-ai to MCP facade; ensure fast-path and LLM routing work. *No voice components yet.*

**Step 1.1 — Refactor Config API** (`pantau/config.py`)

- **Problem:** Current code wraps PantauConfig inside ApplicationConfig; tests import `load_config` and `PantauConfig` directly (fails).
- **Action:** Expose `load_config()` and `PantauConfig` as primary API. Keep ApplicationConfig compatibility if needed for other modules.
- **Current structure (keep):** PantauConfig hierarchy with nested WakeWordConfig, SttConfig, TtsConfig, LlmConfig, HueConfig, HarmonyConfig, SonosConfig, HomekitConfig.
- **Deliverable:** `from pantau.config import PantauConfig, load_config` works; yaml base + env override via `PANTAU_*` and `secret:` prefix.
- **Verify:** `pytest tests/test_config.py` passes; manual load: `cfg = load_config("config/pantau.yaml"); assert cfg.llm.model`

**Step 1.2 — Add Fast-Path DSL Schema & Loader** (`pantau/agent/fast_path.py`, `configuration/fast_path_rules.yaml`)

- **Problem:** Current fast-path logic is a flat `_ROUTES` dictionary and cannot scale to intent templates, aliases, command overrides, or slots.
- **Action:** Define a checked-in DSL with three layers:

  - `intents`: `id`, `templates`, `entity_types`, optional `slots`
  - `entities`: `id`, `type`, `display_name`, `aliases`, optional `default_command`
  - `commands`: `intent`, `entity`, `tool`, `args`, optional `response`

- **Example seed DSL:**

  ```yaml
  intents:
    - id: switch_on
      templates:
        - "schalte {entity} ein"
        - "schalte den {entity} ein"
        - "schalte die {entity} ein"
        - "{entity} einschalten"
        - "mach {entity} an"
        - "mach den {entity} an"
        - "starte {entity}"
      entity_types:
        - device
        - activity
        - light
        - room

    - id: switch_off
      templates:
        - "schalte {entity} aus"
        - "schalte den {entity} aus"
        - "schalte die {entity} aus"
        - "{entity} ausschalten"
        - "mach {entity} aus"
        - "mach den {entity} aus"
        - "stoppe {entity}"
      entity_types:
        - device
        - activity
        - light
        - room

    - id: set_channel
      templates:
        - "schalte auf {entity}"
        - "wechsel zu {entity}"
        - "mach {entity} an"
        - "{entity} einschalten"
      entity_types:
        - tv_channel

    - id: set_volume
      templates:
        - "stelle {entity} auf {value}"
        - "setze {entity} auf {value}"
        - "{entity} auf {value}"
      entity_types:
        - speaker
        - device
      slots:
        value:
          type: integer
          min: 0
          max: 100

  entities:
    - id: fernseher
      type: device
      display_name: "Fernseher"
      aliases:
        - "fernseher"
        - "tv"
        - "glotze"

    - id: wohnzimmer
      type: room
      display_name: "Wohnzimmer"
      aliases:
        - "wohnzimmer"
        - "stube"

    - id: zdf
      type: tv_channel
      display_name: "ZDF"
      aliases:
        - "zdf"
        - "zweite"
        - "zweites"
        - "zweites programm"
      default_command:
        tool: "harmony_set_channel"
        args:
          channel: 2

    - id: ard
      type: tv_channel
      display_name: "ARD"
      aliases:
        - "ard"
        - "erste"
        - "erstes"
        - "das erste"
      default_command:
        tool: "harmony_set_channel"
        args:
          channel: 1

  commands:
    - intent: switch_on
      entity: fernseher
      tool: harmony_start_activity
      args:
        activity: "Fernsehen"
      response: "Ich schalte den Fernseher ein."

    - intent: switch_off
      entity: fernseher
      tool: harmony_stop_activity
      args: {}
      response: "Ich schalte den Fernseher aus."

    - intent: set_channel
      entity: zdf
      tool: harmony_set_channel
      args:
        channel: 2
      response: "Ich schalte auf ZDF."

    - intent: set_channel
      entity: ard
      tool: harmony_set_channel
      args:
        channel: 1
      response: "Ich schalte auf ARD."

    - intent: switch_on
      entity: wohnzimmer
      tool: hue_turn_on_room
      args:
        room: "Wohnzimmer"
      response: "Ich schalte das Wohnzimmerlicht ein."

    - intent: switch_off
      entity: wohnzimmer
      tool: hue_turn_off_room
      args:
        room: "Wohnzimmer"
      response: "Ich schalte das Wohnzimmerlicht aus."
  ```

- **Note:** Use these examples as the seed rule set, but keep the final `tool` names aligned with the actual MCP surface exposed by `pantau/home_mcp/server.py`.

- **Deliverable:** A validated loader that can read and normalize the DSL once at startup and fail fast on invalid definitions.
- **Verify:** Add a focused test that loads the rules and rejects malformed ids, unknown entity types, bad slot definitions, or unreachable command mappings.

**Step 1.3 — Implement Deterministic Intent/Entity Matching** (`pantau/agent/fast_path.py`)

- **Action:** Replace `_ROUTES` with a deterministic matcher pipeline:

  1. normalize text
  2. resolve entity aliases with longest-alias-wins
  3. filter candidate intents by entity type
  4. match templates
  5. parse scalar slots such as `{value}`
  6. resolve command precedence: explicit `commands` entry first, `entity.default_command` second, otherwise no fast-path match

- **Disambiguation rule:** shared templates such as `mach {entity} an` must resolve via entity typing first. Example: `zdf` should resolve to `set_channel`, while `fernseher` should resolve to `switch_on`. If multiple intents remain, use declaration order as the final tiebreaker.
- **Result shape:** keep `fast_path(text) -> FastPathResult | None`, but allow `FastPathResult` to carry `tool`, `args`, and optional metadata such as `intent_id`, `entity_id`, and `response`.
- **Deliverable:** Fast-path recognition works for aliases, article variants, and command overrides without involving the LLM.
- **Verify:** `pytest tests/agent/test_fast_path.py -q`

**Step 1.4 — Rewrite Fast-Path Tests** (`tests/agent/test_fast_path.py`)

- **Action:** Replace `_ROUTES.keys()`-style coverage with DSL-driven tests covering:

  - case and whitespace normalization
  - alias matching (`tv`, `glotze`, `zweites programm`)
  - article/template variants (`schalte den {entity} ein`, `mach {entity} an`)
  - command override precedence over `default_command`
  - default-command fallback
  - slot extraction and range validation for `set_volume`
  - no-match behavior for unrelated phrases
  - overlapping-template disambiguation

- **Deliverable:** Regression coverage for provided examples such as `schalte den fernseher ein`, `mach glotze an`, `schalte auf zweite`, `zdf einschalten`, and `wohnzimmer ausschalten`.
- **Verify:** `pytest tests/agent/test_fast_path.py -q`

**Step 1.5 — Align Prompt and Command Surface** (`pantau/agent/prompts.py`, `pantau/home_mcp/server.py`)

- **Problem:** Current prompt and fast-path examples still mention `pantau_*` wrapper tools, while the repo currently mounts child MCP servers directly.
- **Action:** Pick one command surface and use it consistently in the plan and implementation. Prefer the actual mounted MCP tool names exposed by `pantau/home_mcp/server.py` unless wrapper tools are intentionally reintroduced as a separate task.
- **Deliverable:** Prompt guidance, fast-path examples, and implementation notes no longer mix `pantau_*` pseudo-tools with mounted `harmony_*` / `hue_*` tool families.
- **Verify:** Manual prompt/spec review; no stale wrapper references remain in the updated plan or prompt text.

**Step 1.6 — Update Pydantic-AI Runtime** (`pantau/agent/runtime.py`)

- **Problem:** Uses deprecated `MCPServerStdio`; needs update to current pydantic-ai API.
- **Action:** Research current pydantic-ai MCP composition pattern (check pydantic-ai docs or Context7). Rewrite `build_agent()` to initialize Agent + MCP server correctly.
- **Function signature:** `def build_agent(cfg: PantauConfig) -> Agent`
- **Integration:** Must work with the new `fast_path.py` recognizer and existing `skills.py` modules.
- **Deliverable:** Agent instantiates without warnings; `await agent.run("test text")` executes (e.g., routes to fast-path or LLM).
- **Verify:** No deprecation warnings; manual test with a simple command.

**Step 1.7 — Create Main Entry Point** (`pantau/main.py`)

- **Input flow:** Load config → build agent → accept text command (stdin or test) → process → output response
- **Pipeline:**

  ```text
  text → fast_path recognizer
       → if match: call MCP tool directly using tool + args (+ optional response)
       → else: await agent.run(text)
       → return response
  ```

- **Error handling:** Catch exceptions; log with context; don't crash.
- **Logging:** Use `logger.debug()` for transitions; `logger.info()` for key decisions (routing, tool selection); `logger.warning/error()` for failures.
- **Deliverable:** Executable `python -m pantau.main` that accepts text input and returns a response.
- **Verify:** Manual test — type `"schalte den Fernseher ein"` → fast-path matches → response generated.

**Step 1.8 — Update Dependencies** (`pyproject.toml`)

- **Do NOT add audio packages yet** (that's Phase 2). Only fix any existing dependency conflicts.
- **Verify:** `uv sync` completes; `python -c "from pantau.config import load_config"` works.

**Step 1.9 — Update Config Tests** (`tests/test_config.py`)

- **Current:** Imports fail because `load_config` and `PantauConfig` don't exist.
- **Action:** Update test imports and assertions to match the refactored config API from Step 1.1.
- **Deliverable:** All tests pass.
- **Verify:** `pytest tests/test_config.py -v`

**Phase 1 Success Criteria:**

- ✅ DSL rules load successfully and fail fast on invalid definitions
- ✅ `pytest tests/agent/test_fast_path.py` passes with alias, template, command, and slot coverage
- ✅ `pytest tests/` — all 4+ test modules pass (fast_path, skills, config, MCP server)
- ✅ `python -m pantau.main` accepts text and routes correctly (fast-path or LLM)
- ✅ No pydantic-ai deprecation warnings
- ✅ Config can be loaded: `from pantau.config import load_config; cfg = load_config()`

---

### **Phase 2: Audio Pipeline (4–6 hours)**

**Goal:** Implement all audio components in isolation; verify each works before orchestration.

**Step 2.1 — Add Audio Dependencies** (`pyproject.toml`)

- Add to `[project.dependencies]`:

  ```text
  faster-whisper>=1.0.0
  openwakeword>=0.6.0
  silero-vad>=5.0.0
  piper-tts>=1.2.0
  sounddevice>=0.4.6
  soundfile>=0.12.1
  numpy>=2.0
  torch>=2.3
  ```

- **Verify:** `uv sync`; `python -c "from faster_whisper import WhisperModel; from openwakeword.model import Model"` — no errors.

**Step 2.2 — Implement Silero VAD** (`pantau/audio/vad.py`)

- **API:** `class SileroVAD: def is_speech(chunk: np.ndarray) -> bool:`
- **Purpose:** Pre-STT speech/silence detector; discards non-speech chunks before Whisper to reduce latency.
- **Config:** Initialize once in `__init__`; reuse model.
- **Latency target:** < 1 ms per 30 ms chunk on single CPU.
- **Deliverable:** Working class that can process 16kHz audio chunks.
- **Test:** Unit test — pass silence → `False`, speech → `True`.
- **Reference:** Concept §4.2; use `silero_vad` library directly.

**Step 2.3 — Implement STT** (`pantau/audio/stt.py`)

- **API:** `class GermanSTT: async def record_and_transcribe() -> str:`
- **Config:** `cfg.stt.model_size`, `cfg.stt.device` (auto|cpu|cuda); language always `"de"`.
- **Audio capture:** 16 kHz mono, record until silence detected.

  - Silence threshold: 0.01 RMS
  - Stop after 1.2 s silence or 10 s total

- **Whisper settings:** `beam_size=1`, `vad_filter=True`, `condition_on_previous_text=False`
- **Compute type:** `int8` (CPU) or `float16` (CUDA) — select based on device.
- **Deliverable:** Async function that captures audio and returns transcribed German text.
- **Latency target:** ~1 s per 3 s of audio (small model on CPU).
- **Test:** Manual — speak German, verify accurate transcription within target time.
- **Reference:** Concept §4.3; use `faster_whisper` library.

**Step 2.4 — Implement Wake-Word Detector** (`pantau/audio/wakeword.py`)

- **API:** `class WakeWordListener: async def listen() -> None:` — blocks until wake word detected.
- **Config:** `cfg.wake_word.model` (path to `.tflite`), `cfg.wake_word.threshold`, `cfg.wake_word.post_wake_timeout_s`.
- **Audio input:** 16 kHz mono, 80 ms chunks (OpenWakeWord standard).
- **Detection:** Load model with `openwakeword.model.Model`; run prediction on each chunk; fire event when confidence ≥ threshold.
- **Latency target:** < 100 ms from wake word utterance to detection callback.
- **MVP:** Use placeholder model or existing pretrained model; custom "Pantau" training deferred.
- **Deliverable:** Async function that blocks until wake word detected.
- **Test:** Say "hey jarvis" or placeholder phrase → callback fires within 100 ms.
- **Reference:** Concept §4.1; use `openwakeword` library.

**Step 2.5 — Implement TTS** (`pantau/audio/tts.py`)

- **API:** `class PiperTTS: async def speak(text: str) -> None:`
- **Config:** `cfg.tts.model` (path to ONNX), `cfg.tts.speak_rate`.
- **Model:** `de_DE-thorsten-high` voice; load once in `__init__`.
- **Synthesis + playback:** Generate WAV from text; stream to speaker in real-time.
- **Latency target:** < 400 ms synthesis on CPU.
- **Deliverable:** Async function that speaks German text via speaker.
- **Test:** Manual — `await tts.speak("Hallo Welt")` → audio plays.
- **Reference:** Concept §4.4; use `piper` library.

**Phase 2 Success Criteria:**

- ✅ All audio classes instantiate without import errors
- ✅ Wake-word detected in < 100 ms
- ✅ STT transcribes 3–5 s German speech in < 2 s (verify with manual test)
- ✅ TTS speaks back a response (audio audible)
- ✅ Unit tests for each component pass
- ✅ No resource leaks (audio streams cleaned up after use)

---

### **Phase 3: End-to-End Voice Orchestration (2–3 hours)**

**Goal:** Wire all components; validate full voice interaction loop end-to-end.

**Step 3.1 — Update main.py with Full Voice Pipeline** (`pantau/main.py`)

- **Async orchestration:**

  ```python
  async def main():
      cfg = load_config()
      wakeword = WakeWordListener(cfg.wake_word)
      stt = GermanSTT(cfg.stt)
      tts = PiperTTS(cfg.tts)
      agent = build_agent(cfg)

      await tts.speak("Pantau ist bereit.")
      logger.info("Ready; waiting for wake word...")

      while True:
          try:
              await wakeword.listen()
              await tts.speak("Ja?")

              text = await stt.record_and_transcribe()
              if not text:
                  continue
              logger.info("Recognized: %s", text)

              # Fast-path recognizer
              match = fast_path(text)
              if match:
                  logger.info("Fast-path: %s %s", match.tool, match.args)
                  response = match.response or await call_mcp_tool(match.tool, match.args)
              else:
                  async with agent.run_mcp_servers():
                      result = await agent.run(text)
                  response = result.output

              logger.info("Response: %s", response)
              await tts.speak(response)
          except Exception:
              logger.exception("Unhandled error")
              await tts.speak("Es gab einen Fehler. Bitte versuche es erneut.")
  ```

- **Error handling:** Catch all exceptions; speak error message; continue loop (no crash).
- **Logging:** Log every transition (detected text, routing decision, MCP tool called, response).
- **Deliverable:** Complete voice loop executable.
- **Verify:** Full conversation test — say "schalte das Wohnzimmer ein" → hear response; confirm light turns on (if hardware available).

**Step 3.2 — Integration Tests** (in `tests/` or `evals/`)

- **Test fast-path routing:** Ensure main.py triggers fast-path for known phrases (not already tested elsewhere).
- **Test LLM routing:** Mock agent response; verify MCP tool called with correct args.
- **Reference:** Existing `evals/commands_de.yaml` provides test cases.
- **Deliverable:** Integration tests that cover fast-path + LLM paths.
- **Verify:** `pytest tests/ -k integration` (or similar) passes.

**Step 3.3 — Latency Validation** (measurement, not code)

- **Profile each stage:** Measure wake-word detection, STT, LLM round-trip, TTS.
- **Compare to spec targets** (Concept §8):

  - Fast-path total: ~2–4 s ✅
  - LLM path total: ~2.5–5 s ✅

- **If over budget:** Profile and optimize (e.g., STT slow? try `tiny` model; LLM slow? check network round-trip).
- **Log results:** Document actual latency vs. targets in task notes.

**Phase 3 Success Criteria:**

- ✅ Full voice loop works: wake → listen → respond → TTS
- ✅ Fast-path commands execute within 2–4 s latency
- ✅ LLM commands within 2.5–5 s
- ✅ Graceful error handling; no crashes mid-interaction
- ✅ All integration tests pass

---

### **Phase 4: Production Hardening (Optional Stretch — 3–5 hours)**

**Goal:** Robustness, observability, fallback handling.

**Steps (choose based on priority):**

1. **Audit logging** — Ensure every tool call is logged with args/result; full stack traces on error.
2. **Device fallbacks** — Handle unreachable devices gracefully; speak user-friendly error message.
3. **Confirmation prompts** — Risky actions (e.g., "turn off all lights") ask for confirmation before execution.
4. **Per-stage metrics** — Log latency of each audio/agent stage for performance tuning.
5. **Barge-in support** — Allow user to interrupt during TTS playback (if feasible with sounddevice).
6. **Startup validation** — On app start, check that all configured devices are reachable; warn if not.

---

## Technical Decisions & Rationale

**Audio Stack:**

| Component | Choice | Rationale |
| --- | --- | --- |
| **Wake word** | OpenWakeWord | Local, trainable, Apache 2.0; custom "Pantau" model available; Porcupine requires cloud dependency |
| **VAD** | Silero VAD | < 1 ms per chunk; local; accurate speech/silence boundary; reduces STT load |
| **STT** | faster-whisper `small` | Local, German-capable, ~1 s per 3 s audio on CPU; int8 compression for RAM efficiency; no cloud |
| **TTS** | Piper TTS | < 400 ms synthesis; ONNX-based; `thorsten-high` voice natural; local |

**Agent & MCP:**

- **pydantic-ai** (not pydantic-deepagents) — Pantau is reactive (1 utterance → 1 tool), not multi-step planning; deepagents overhead unjustified.
- **FastMCP facade (7 tools)** — Fewer tools = faster, more reliable LLM routing; security perimeter; avoids exposing low-level device primitives.
- **Fast-path recognizer = deterministic DSL, not ML** — keep latency low by compiling templates, aliases, and command mappings once rather than using a classifier or NER model.
- **gpt-5.4-nano MVP** — $0.20/1M tokens; negligible cost per command; swap to Ollama + `qwen3` offline with one config line.

**Configuration:**

- **YAML base + env override** — Standard appkit pattern; `secret:` prefix for Key Vault integration (prod).
- **Nested config objects** — Per-stage configs (WakeWordConfig, SttConfig, etc.) for clean composition and testing.
- **No f-strings in logger calls** — Per appkit conventions; use `logger.info("msg %s", var)` not f-strings.

---

## Critical Notes for Implementation

- **Async throughout.** All audio and agent operations are async. Use `asyncio.to_thread()` for blocking I/O (e.g., sounddevice operations).
- **Resource cleanup.** Audio streams must be properly closed. Test with `asyncio.run()` to catch resource leaks.
- **Logging style.** No f-strings in logger calls. Use: `logger.info("Recognized: %s", text)` not `f"Recognized: {text}"`.
- **Testing approach.** Unit test fast-path DSL loading and matching before broader runtime work. Unit test each audio class in isolation (mock sounddevice if needed). Integration test the full pipeline with a mock MCP server. Finish with a manual end-to-end voice acceptance test.
- **Latency tuning.** If STT too slow, downgrade to `tiny` model. If LLM too slow, check network round-trip or use local Ollama.
- **Device knowledge.** `pantau/skills/hue_rooms.md` and `harmony_activities.md` document room/activity mappings for LLM context.
- **Fast-path scope.** Keep fast-path deterministic: templates, aliases, explicit command mappings, default-command fallback, and scalar slots such as `{value}` are in scope; fuzzy matching and confidence scoring are not.

---

## Verification Checklist

### Phase 1 ✅

- [ ] Fast-path DSL rules load and validate successfully
- [ ] Alias, template, command override, and slot tests pass in `tests/agent/test_fast_path.py`
- [ ] `pytest tests/` passes (all test modules)
- [ ] `from pantau.config import PantauConfig, load_config` succeeds
- [ ] `python -m pantau.main` + text input → routes and responds
- [ ] No pydantic-ai deprecation warnings

### Phase 2 ✅

- [ ] Audio classes instantiate without import errors
- [ ] Wake-word detected in < 100 ms
- [ ] STT transcribes 3–5 s German audio in < 2 s
- [ ] TTS speaks response (audible on speaker)
- [ ] Unit tests for audio components pass

### Phase 3 ✅

- [ ] Full voice loop: wake → listen → respond → TTS
- [ ] Latency: fast-path 2–4 s, LLM 2.5–5 s
- [ ] Graceful error handling; no crashes

### Phase 4 (Optional) ✅

- [ ] All tool calls logged with args/result
- [ ] Unreachable devices handled gracefully
- [ ] Risky actions confirm before execution
- [ ] Per-stage latency tracked

---

## References

- **Architecture & rationale:** [pantau-concept-v2.md](pantau-concept-v2.md)
- **Working code to reference/extend:**
  - Fast-path: `pantau/agent/fast_path.py`
  - Fast-path tests: `tests/agent/test_fast_path.py`
  - Fast-path prompt guidance: `pantau/agent/prompts.py`
  - MCP facade: `pantau/home_mcp/server.py`
  - Config: `pantau/config.py`
  - Skills: `pantau/skills/`
- **Test patterns:** `tests/agent/test_fast_path.py`, `tests/home_mcp/test_server.py`

---

## Timeline & Effort Estimate

| Phase | Duration | Notes |
| --- | --- | --- |
| 1 | 2–3 h | Config refactor + pydantic-ai fix + main.py |
| 2 | 4–6 h | Audio stack (heaviest lifting) |
| 3 | 2–3 h | Integration + latency tuning |
| **Total (1–3)** | **~8–12 h** | Core voice agent complete |
| 4 (optional) | 3–5 h | Production hardening |

---

## Handoff Checklist for Developer

Before starting implementation:

- [ ] Read `pantau-concept-v2.md` in full
- [ ] Understand current state & gaps (above)
- [ ] Review appkit conventions (async, logging, config API)
- [ ] Verify audio test environment (speakers/mic functional)
- [ ] Run `uv sync` before starting
- [ ] Review existing test patterns in `tests/agent/` and `tests/home_mcp/`

During implementation:

- [ ] Start with Phase 1; don't add audio until text-only agent works
- [ ] Treat fast-path redesign as a prerequisite for runtime integration, not a follow-up cleanup
- [ ] After each step, run `pytest` + manual test
- [ ] Log all decisions, blockers, learnings to pantau project memory
- [ ] Commit frequently with descriptive messages
