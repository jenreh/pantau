from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FastPathResult:
    tool: str
    args: dict[str, object]


_ROUTES: dict[str, FastPathResult] = {
    phrase: result
    for phrases, result in [
        (
            {
                "schalte den fernseher ein",
                "mach den fernseher an",
                "fernseher ein",
                "tv ein",
                "tv an",
            },
            FastPathResult("pantau_start_tv", {"activity": "Fernsehen"}),
        ),
        (
            {"schalte den fernseher aus", "fernseher aus", "tv aus"},
            FastPathResult("pantau_power_off_tv", {}),
        ),
        (
            {
                "schalte das wohnzimmer ein",
                "wohnzimmer ein",
                "licht im wohnzimmer an",
                "wohnzimmer licht an",
            },
            FastPathResult("pantau_turn_on_room", {"room": "Wohnzimmer"}),
        ),
        (
            {"schalte das wohnzimmer aus", "wohnzimmer aus", "licht im wohnzimmer aus"},
            FastPathResult("pantau_turn_off_room", {"room": "Wohnzimmer"}),
        ),
    ]
    for phrase in phrases
}


def fast_path(text: str) -> FastPathResult | None:
    return _ROUTES.get(text.lower().strip())
