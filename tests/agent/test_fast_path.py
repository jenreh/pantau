from __future__ import annotations

from pathlib import Path

import pytest

from pantau.agent.fast_path import _match, _Rules, load_rules


@pytest.fixture(scope="module")
def rules() -> _Rules:
    return load_rules()


# --- TV (Fernseher) ---


def test_tv_on_full_phrase(rules: _Rules) -> None:
    r = _match("schalte den fernseher ein", rules)
    assert r is not None
    assert r.tool == "harmony_start_activity"
    assert r.args == {"activity": "Fernsehen"}
    assert r.intent_id == "switch_on"
    assert r.entity_id == "fernseher"


def test_tv_on_alias_tv(rules: _Rules) -> None:
    r = _match("tv ein", rules)
    assert r is not None
    assert r.tool == "harmony_start_activity"


def test_tv_on_alias_glotze(rules: _Rules) -> None:
    r = _match("mach glotze an", rules)
    assert r is not None
    assert r.tool == "harmony_start_activity"


def test_tv_off(rules: _Rules) -> None:
    r = _match("schalte den fernseher aus", rules)
    assert r is not None
    assert r.tool == "harmony_power_off"
    assert r.response == "Ich schalte den Fernseher aus."


def test_tv_off_short(rules: _Rules) -> None:
    r = _match("fernseher ausschalten", rules)
    assert r is not None
    assert r.tool == "harmony_power_off"


# --- Room (Wohnzimmer) ---


def test_room_on(rules: _Rules) -> None:
    r = _match("schalte das wohnzimmer ein", rules)
    assert r is not None
    assert r.tool == "hue_set_room_on"
    assert r.args["room"] == "Wohnzimmer"
    assert r.args["on"] is True


def test_room_off(rules: _Rules) -> None:
    r = _match("wohnzimmer ausschalten", rules)
    assert r is not None
    assert r.tool == "hue_set_room_on"
    assert r.args["on"] is False


def test_room_alias_stube(rules: _Rules) -> None:
    r = _match("mach die stube an", rules)
    assert r is not None
    assert r.tool == "hue_set_room_on"
    assert r.args["on"] is True


def test_room_light_phrase_routes_to_room_on(rules: _Rules) -> None:
    r = _match("licht im wohnzimmer an", rules)
    assert r is not None
    assert r.tool == "hue_set_room_on"
    assert r.args == {"room": "Wohnzimmer", "on": True}


def test_room_light_phrase_flur_short_imperative(rules: _Rules) -> None:
    r = _match("schalt das licht im flur ein", rules)
    assert r is not None
    assert r.tool == "hue_set_room_on"
    assert r.args == {"room": "Flur", "on": True}
    assert r.intent_id == "switch_on"
    assert r.entity_id == "flur"


# --- Channels (ZDF / ARD) ---


def test_zdf_channel(rules: _Rules) -> None:
    r = _match("schalte auf zdf", rules)
    assert r is not None
    assert r.tool == "harmony_set_channel"
    assert r.args["channel"] == "2"
    assert r.intent_id == "set_channel"


def test_zdf_alias_zweite(rules: _Rules) -> None:
    r = _match("schalte auf zweite", rules)
    assert r is not None
    assert r.tool == "harmony_set_channel"
    assert r.args["channel"] == "2"


def test_zdf_alias_zweites_programm(rules: _Rules) -> None:
    r = _match("schalte auf zweites programm", rules)
    assert r is not None
    assert r.tool == "harmony_set_channel"


def test_ard_default_command_fallback(rules: _Rules) -> None:
    # ARD has no explicit command entry — falls back to entity.default_command
    r = _match("schalte auf ard", rules)
    assert r is not None
    assert r.tool == "harmony_set_channel"
    assert r.args["channel"] == "1"


def test_ard_alias_das_erste(rules: _Rules) -> None:
    r = _match("mach das erste an", rules)
    assert r is not None
    assert r.tool == "harmony_set_channel"


# --- Disambiguation ---


def test_zdf_einschalten_routes_to_set_channel_not_switch_on(rules: _Rules) -> None:
    r = _match("zdf einschalten", rules)
    assert r is not None
    assert r.intent_id == "set_channel"
    assert r.tool == "harmony_set_channel"


# --- Slots ---


def test_volume_slot_extraction(rules: _Rules) -> None:
    r = _match("stelle lautsprecher auf 50", rules)
    assert r is not None
    assert r.tool == "sonos_set_volume"
    assert r.args["value"] == 50


def test_volume_slot_range_invalid_above_max(rules: _Rules) -> None:
    r = _match("stelle lautsprecher auf 150", rules)
    assert r is None


def test_volume_slot_range_min(rules: _Rules) -> None:
    r = _match("stelle lautsprecher auf 0", rules)
    assert r is not None
    assert r.args["value"] == 0


# --- Normalization ---


def test_case_insensitive(rules: _Rules) -> None:
    r1 = _match("TV EIN", rules)
    r2 = _match("tv ein", rules)
    assert r1 is not None
    assert r2 is not None
    assert r1.tool == r2.tool


def test_leading_trailing_whitespace(rules: _Rules) -> None:
    r = _match("  schalte den fernseher ein  ", rules)
    assert r is not None
    assert r.tool == "harmony_start_activity"


def test_extra_internal_whitespace(rules: _Rules) -> None:
    r = _match("schalte  den  fernseher  ein", rules)
    assert r is not None
    assert r.tool == "harmony_start_activity"


# --- No match ---


def test_no_match_unknown_phrase(rules: _Rules) -> None:
    assert _match("spiele jazz im wohnzimmer", rules) is None


def test_no_match_empty_string(rules: _Rules) -> None:
    assert _match("", rules) is None


def test_no_match_whitespace_only(rules: _Rules) -> None:
    assert _match("   ", rules) is None


def test_no_match_partial_phrase(rules: _Rules) -> None:
    assert _match("schalte", rules) is None


# --- DSL validation ---


def test_load_rules_returns_rules_object(rules: _Rules) -> None:
    assert isinstance(rules, _Rules)
    assert len(rules.intents) > 0
    assert len(rules.entities) > 0


def test_load_rules_rejects_duplicate_intent_id(tmp_path: Path) -> None:
    import yaml

    bad_yaml = yaml.dump(
        {
            "intents": [
                {"id": "switch_on", "templates": ["a"], "entity_types": ["device"]},
                {"id": "switch_on", "templates": ["b"], "entity_types": ["room"]},
            ],
            "entities": [],
            "commands": [],
        }
    )
    p = tmp_path / "bad.yaml"
    p.write_text(bad_yaml, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate intent id"):
        load_rules(p)


def test_fast_path_response_field(rules: _Rules) -> None:
    r = _match("schalte den fernseher ein", rules)
    assert r is not None
    assert r.response == "Ich schalte den Fernseher ein."
