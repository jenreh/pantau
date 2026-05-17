from __future__ import annotations

from pathlib import Path

from pantau.agent.skills import load_skills


def test_load_skills_returns_joined_markdown(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Section A", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Section B", encoding="utf-8")

    result = load_skills(tmp_path)

    assert "# Section A" in result
    assert "# Section B" in result
    assert "---" in result


def test_load_skills_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "z.md").write_text("Z content", encoding="utf-8")
    (tmp_path / "a.md").write_text("A content", encoding="utf-8")

    result = load_skills(tmp_path)

    assert result.index("A content") < result.index("Z content")


def test_load_skills_single_file_no_separator(tmp_path: Path) -> None:
    (tmp_path / "only.md").write_text("Only section", encoding="utf-8")

    result = load_skills(tmp_path)

    assert result == "Only section"
    assert "---" not in result


def test_load_skills_empty_dir_returns_empty_string(tmp_path: Path) -> None:
    assert load_skills(tmp_path) == ""
