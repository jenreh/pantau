from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_RULES_PATH = (
    Path(__file__).parent.parent.parent / "configuration" / "fast_path_rules.yaml"
)


@dataclass
class FastPathResult:
    tool: str
    args: dict[str, object]
    intent_id: str = ""
    entity_id: str = ""
    response: str | None = None


@dataclass
class _SlotDef:
    type: str
    min: int | None = None
    max: int | None = None


@dataclass
class _CommandDef:
    tool: str
    args: dict[str, object]
    response: str | None = None


@dataclass
class _IntentDef:
    id: str
    templates: list[str]
    entity_types: list[str]
    slots: dict[str, _SlotDef] = field(default_factory=dict)


@dataclass
class _EntityDef:
    id: str
    type: str
    display_name: str
    aliases: list[str]
    default_command: _CommandDef | None = None


class _Rules:
    def __init__(
        self,
        intents: list[_IntentDef],
        entities: list[_EntityDef],
        commands: dict[tuple[str, str], _CommandDef],
    ) -> None:
        self.intents = intents
        self.entities = entities
        self.commands = commands
        self.entity_by_id: dict[str, _EntityDef] = {e.id: e for e in entities}
        self.alias_index: list[tuple[str, str]] = sorted(
            [(alias, e.id) for e in entities for alias in e.aliases],
            key=lambda pair: len(pair[0]),
            reverse=True,
        )


def _parse_command_def(raw: dict) -> _CommandDef:
    return _CommandDef(
        tool=raw["tool"],
        args=raw.get("args") or {},
        response=raw.get("response"),
    )


def load_rules(path: Path = _RULES_PATH) -> _Rules:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    intent_ids: set[str] = set()
    intents: list[_IntentDef] = []
    for item in data.get("intents", []):
        id_ = item["id"]
        if id_ in intent_ids:
            raise ValueError(f"Duplicate intent id: {id_!r}")
        intent_ids.add(id_)
        slots: dict[str, _SlotDef] = {}
        for name, s in item.get("slots", {}).items():
            slots[name] = _SlotDef(type=s["type"], min=s.get("min"), max=s.get("max"))
        intents.append(
            _IntentDef(
                id=id_,
                templates=item["templates"],
                entity_types=item["entity_types"],
                slots=slots,
            )
        )

    entity_ids: set[str] = set()
    entities: list[_EntityDef] = []
    for item in data.get("entities", []):
        id_ = item["id"]
        if id_ in entity_ids:
            raise ValueError(f"Duplicate entity id: {id_!r}")
        entity_ids.add(id_)
        dc = item.get("default_command")
        entities.append(
            _EntityDef(
                id=id_,
                type=item["type"],
                display_name=item["display_name"],
                aliases=[a.lower() for a in item["aliases"]],
                default_command=_parse_command_def(dc) if dc else None,
            )
        )

    commands: dict[tuple[str, str], _CommandDef] = {}
    for item in data.get("commands", []):
        intent_id = item["intent"]
        entity_id = item["entity"]
        if intent_id not in intent_ids:
            raise ValueError(f"Command references unknown intent: {intent_id!r}")
        if entity_id not in entity_ids:
            raise ValueError(f"Command references unknown entity: {entity_id!r}")
        commands[(intent_id, entity_id)] = _parse_command_def(item)

    logger.debug(
        "Loaded %d intents, %d entities, %d commands from %s",
        len(intents),
        len(entities),
        len(commands),
        path,
    )
    return _Rules(intents, entities, commands)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _find_entity(text: str, rules: _Rules) -> tuple[str, str] | None:
    for alias, entity_id in rules.alias_index:
        if alias in text:
            return alias, entity_id
    return None


def _template_matches(template: str, text_with_entity: str) -> tuple[bool, int | None]:
    if "{value}" not in template:
        return text_with_entity == template, None

    parts = template.split("{value}")
    if len(parts) != 2:
        return False, None
    prefix, suffix = parts
    if not text_with_entity.startswith(prefix):
        return False, None
    remainder = text_with_entity[len(prefix) :]
    if suffix:
        if not remainder.endswith(suffix):
            return False, None
        value_str = remainder[: -len(suffix)]
    else:
        value_str = remainder
    if not value_str.strip().isdigit():
        return False, None
    return True, int(value_str.strip())


def _match(text: str, rules: _Rules) -> FastPathResult | None:
    normalized = _normalize(text)
    if not normalized:
        return None

    found = _find_entity(normalized, rules)
    if found is None:
        return None
    matched_alias, entity_id = found
    entity = rules.entity_by_id[entity_id]

    text_with_placeholder = normalized.replace(matched_alias, "{entity}", 1)

    candidate_intents = [i for i in rules.intents if entity.type in i.entity_types]

    matched_intent: _IntentDef | None = None
    slots: dict[str, object] = {}

    for intent in candidate_intents:
        for template in intent.templates:
            ok, value = _template_matches(template, text_with_placeholder)
            if not ok:
                continue
            if value is not None:
                slot = intent.slots.get("value")
                if slot:
                    if slot.min is not None and value < slot.min:
                        continue
                    if slot.max is not None and value > slot.max:
                        continue
                slots["value"] = value
            matched_intent = intent
            break
        if matched_intent:
            break

    if matched_intent is None:
        return None

    command = rules.commands.get((matched_intent.id, entity_id))
    if command is None:
        command = entity.default_command
    if command is None:
        return None

    final_args = {**command.args, **slots}

    return FastPathResult(
        tool=command.tool,
        args=final_args,
        intent_id=matched_intent.id,
        entity_id=entity_id,
        response=command.response,
    )


_cached_rules: _Rules | None = None


def _get_rules() -> _Rules:
    global _cached_rules
    if _cached_rules is None:
        _cached_rules = load_rules()
    return _cached_rules


def fast_path(text: str) -> FastPathResult | None:
    return _match(text, _get_rules())
