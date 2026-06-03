"""Tests for the shared ingestor cell I/O helpers (tm.utils).

existing_articles() and save_article() were factored out of the four batch
ingestors. These pin the cache-check glob, the zero-padded article naming, the
dir-creation side effect, and the non-ASCII-preserving JSON dump.
"""

import json

from tm.utils import existing_articles, save_article


class TestExistingArticles:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert existing_articles(tmp_path / "nope") == []

    def test_lists_only_article_json(self, tmp_path):
        (tmp_path / "article_01.json").write_text("{}")
        (tmp_path / "article_02.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "meta.json").write_text("{}")
        found = existing_articles(tmp_path)
        assert sorted(p.name for p in found) == ["article_01.json", "article_02.json"]


class TestSaveArticle:
    def test_creates_dir_and_writes_padded_name(self, tmp_path):
        cell = tmp_path / "gdelt" / "C05"
        out = save_article(cell, 1, {"headline": "h", "text": "t"})
        assert out == cell / "article_01.json"
        assert out.exists()
        assert json.loads(out.read_text()) == {"headline": "h", "text": "t"}

    def test_two_digit_index(self, tmp_path):
        out = save_article(tmp_path, 12, {"x": 1})
        assert out.name == "article_12.json"

    def test_preserves_non_ascii(self, tmp_path):
        out = save_article(tmp_path, 3, {"headline": "מבחן", "text": "עברית"})
        raw = out.read_text()
        assert "מבחן" in raw  # ensure_ascii=False keeps the original glyphs
        assert json.loads(raw)["text"] == "עברית"

    def test_returned_path_is_consistent_with_existing_articles(self, tmp_path):
        save_article(tmp_path, 1, {"a": 1})
        save_article(tmp_path, 2, {"b": 2})
        assert len(existing_articles(tmp_path)) == 2
