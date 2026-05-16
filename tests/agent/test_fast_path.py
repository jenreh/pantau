from __future__ import annotations

import pytest

from pantau.agent.fast_path import FastPathResult, _ROUTES, fast_path


def test_fast_path_tv_on() -> None:
    result = fast_path("TV ein")
    assert result is not None
    assert result.tool == "pantau_start_tv"
    assert result.args == {"activity": "Fernsehen"}


def test_fast_path_tv_off() -> None:
    result = fast_path("fernseher aus")
    assert result is not None
    assert result.tool == "pantau_power_off_tv"


def test_fast_path_wohnzimmer_on() -> None:
    result = fast_path("Wohnzimmer ein")
    assert result is not None
    assert result.tool == "pantau_turn_on_room"
    assert result.args["room"] == "Wohnzimmer"


def test_fast_path_wohnzimmer_off() -> None:
    result = fast_path("wohnzimmer aus")
    assert result is not None
    assert result.tool == "pantau_turn_off_room"


def test_fast_path_case_insensitive() -> None:
    assert fast_path("TV EIN") == fast_path("tv ein")


def test_fast_path_strips_whitespace() -> None:
    result = fast_path("  tv ein  ")
    assert result is not None
    assert result.tool == "pantau_start_tv"


def test_fast_path_unknown_returns_none() -> None:
    assert fast_path("spiele jazz im wohnzimmer") is None
    assert fast_path("") is None
    assert fast_path("   ") is None


@pytest.mark.parametrize("phrase", list(_ROUTES.keys()))
def test_all_routes_return_result(phrase: str) -> None:
    result = fast_path(phrase)
    assert isinstance(result, FastPathResult)
    assert result.tool.startswith("pantau_")
