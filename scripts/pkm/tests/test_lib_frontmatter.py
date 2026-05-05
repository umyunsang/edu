import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib_frontmatter import merge_frontmatter, read_note, write_note  # noqa: E402

FIX = pathlib.Path(__file__).parent / "fixtures"


def test_read_note_with_frontmatter():
    fm, body = read_note(FIX / "sample_with_fm.md")
    assert fm["title"] == "머신러닝 기초"
    assert fm["tags"] == ["ML"]
    assert "[[관련노트]]" in body


def test_read_note_without_frontmatter():
    fm, body = read_note(FIX / "sample_no_fm.md")
    assert fm == {}
    assert body.startswith("# 무제 노트")


def test_merge_preserves_existing_keys():
    base = {"title": "원래", "tags": ["A"]}
    new = {"tags": ["A", "B"], "type": "lecture"}
    merged = merge_frontmatter(base, new)
    assert merged["title"] == "원래"
    assert sorted(merged["tags"]) == ["A", "B"]
    assert merged["type"] == "lecture"


def test_merge_does_not_overwrite_protected():
    base = {"title": "원래", "date": "2026-01-01"}
    new = {"title": "새것", "date": "2099-12-31"}
    merged = merge_frontmatter(base, new, protected=["title", "date"])
    assert merged["title"] == "원래"
    assert merged["date"] == "2026-01-01"


def test_write_roundtrip(tmp_path):
    fm = {"title": "테스트", "tags": ["x"]}
    body = "# Hi\n\n본문"
    p = tmp_path / "test.md"
    write_note(p, fm, body)
    fm2, body2 = read_note(p)
    assert fm2 == fm
    assert body2.strip() == body.strip()
