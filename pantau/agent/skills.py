from __future__ import annotations

from pathlib import Path


def load_skills(skills_dir: Path = Path("pantau/skills")) -> str:
    sections = [f.read_text(encoding="utf-8") for f in sorted(skills_dir.glob("*.md"))]
    return "\n\n---\n\n".join(sections)
