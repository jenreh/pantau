from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart

from pantau.agent.fast_path import FastPathResult
from pantau.config import ApplicationConfig, LlmConfig, McpConfig, McpServerConfig
from pantau.session import PantauSession, process


@pytest.mark.asyncio
async def test_process_executes_fast_path_tool_and_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(),
        mcp=McpConfig(
            servers=[McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"])],
        ),
    )
    executed: list[tuple[list[str], str, dict[str, object]]] = []

    class FakeRegistry:
        def get(self, config_type: type[ApplicationConfig]) -> ApplicationConfig:
            assert config_type is ApplicationConfig
            return cfg

    class FakeAgent:
        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    async def fake_execute_mcp_tool(
        servers: list[McpServerConfig],
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, str]:
        executed.append(([server.name for server in servers], tool_name, arguments))
        return {"status": "ok"}

    monkeypatch.setattr("pantau.session.service_registry", FakeRegistry)
    monkeypatch.setattr(
        "pantau.session.fast_path",
        lambda text: FastPathResult(
            tool="hue_set_room_on",
            args={"room": "Flur", "on": True},
            intent_id="switch_on",
            entity_id="flur",
            response="Ich schalte das Flurlicht ein.",
        ),
    )
    monkeypatch.setattr("pantau.session.execute_mcp_tool", fake_execute_mcp_tool)
    monkeypatch.setattr(
        "pantau.session.resolve_available_mcp_servers", lambda servers: servers
    )
    monkeypatch.setattr(
        "pantau.session.build_agent", lambda app_cfg, mcp_servers: FakeAgent()
    )

    result = await process("schalte das licht im flur ein")

    assert result == "Ich schalte das Flurlicht ein."
    assert executed == [
        (["huehub"], "hue_set_room_on", {"room": "Flur", "on": True}),
    ]


@pytest.mark.asyncio
async def test_session_reuses_agent_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(),
        mcp=McpConfig(
            servers=[McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"])],
        ),
    )
    build_calls: list[list[str]] = []

    class FakeRegistry:
        def get(self, config_type: type[ApplicationConfig]) -> ApplicationConfig:
            assert config_type is ApplicationConfig
            return cfg

    class FakeAgent:
        def __init__(self) -> None:
            self.run_calls: list[str] = []
            self.received_message_history: list[list[object]] = []
            self.enter_calls = 0

        async def __aenter__(self) -> FakeAgent:
            self.enter_calls += 1
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def run(
            self,
            text: str,
            message_history: list[object] | None = None,
        ) -> SimpleNamespace:
            self.run_calls.append(text)
            self.received_message_history.append(list(message_history or []))
            prior_messages = list(message_history or [])
            return SimpleNamespace(
                output=f"done:{text}",
                all_messages=lambda: [*prior_messages, f"history:{text}"],
            )

    fake_agent = FakeAgent()

    monkeypatch.setattr(
        "pantau.session.service_registry",
        FakeRegistry,
    )
    monkeypatch.setattr("pantau.session.fast_path", lambda text: None)
    monkeypatch.setattr(
        "pantau.session.resolve_available_mcp_servers",
        lambda servers: servers,
    )
    monkeypatch.setattr(
        "pantau.session.build_agent",
        lambda app_cfg, mcp_servers: (
            build_calls.append([s.name for s in mcp_servers]) or fake_agent
        ),
    )

    async with PantauSession() as session:
        first = await process("eins", session=session)
        second = await process("zwei", session=session)

    assert first == "done:eins"
    assert second == "done:zwei"
    assert build_calls == [["huehub"]]
    assert fake_agent.enter_calls == 1
    assert fake_agent.run_calls == ["eins", "zwei"]
    assert fake_agent.received_message_history == [[], ["history:eins"]]


@pytest.mark.asyncio
async def test_session_records_fast_path_turns_for_later_llm_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ApplicationConfig(
        version="0.1.0",
        name="pantau",
        logging="logging.yaml",
        llm=LlmConfig(),
        mcp=McpConfig(
            servers=[McpServerConfig(name="huehub", args=["-m", "huehub.mcp_server"])],
        ),
    )

    class FakeRegistry:
        def get(self, config_type: type[ApplicationConfig]) -> ApplicationConfig:
            assert config_type is ApplicationConfig
            return cfg

    class FakeAgent:
        def __init__(self) -> None:
            self.received_message_history: list[list[object]] = []

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def run(
            self,
            text: str,
            message_history: list[object] | None = None,
        ) -> SimpleNamespace:
            self.received_message_history.append(list(message_history or []))
            return SimpleNamespace(
                output=f"done:{text}", all_messages=lambda: list(message_history or [])
            )

    fake_agent = FakeAgent()
    fast_path_results = iter(
        [
            FastPathResult(
                tool="hue_set_light_on",
                args={"light": "Flurlampe Tür", "on": True},
                intent_id="switch_on",
                entity_id="tuer_licht_1",
                response="Okay, das Türlicht \u201eFlurlampe Tür\u201c ist jetzt eingeschaltet.",
            ),
            None,
        ]
    )

    async def fake_execute_mcp_tool(
        servers: list[McpServerConfig],
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, str]:
        return {"status": "ok"}

    monkeypatch.setattr("pantau.session.service_registry", FakeRegistry)
    monkeypatch.setattr(
        "pantau.session.fast_path", lambda text: next(fast_path_results)
    )
    monkeypatch.setattr("pantau.session.execute_mcp_tool", fake_execute_mcp_tool)
    monkeypatch.setattr(
        "pantau.session.resolve_available_mcp_servers",
        lambda servers: servers,
    )
    monkeypatch.setattr(
        "pantau.session.build_agent", lambda app_cfg, mcp_servers: fake_agent
    )

    async with PantauSession() as session:
        first = await process("schalte das tür licht 1 ein", session=session)
        second = await process("schalte die lampe wieder aus", session=session)

    assert first == "Okay, das Türlicht „Flurlampe Tür“ ist jetzt eingeschaltet."
    assert second == "done:schalte die lampe wieder aus"

    history = fake_agent.received_message_history[0]
    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[1], ModelResponse)
    assert history[0].parts[0].content == "schalte das tür licht 1 ein"
    assert isinstance(history[1].parts[0], TextPart)
    assert (
        history[1].parts[0].content
        == "Okay, das Türlicht „Flurlampe Tür“ ist jetzt eingeschaltet."
    )
