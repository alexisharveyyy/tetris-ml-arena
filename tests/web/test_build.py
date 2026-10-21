"""The committed frontend build must match web/src (rebuild with npm run build)."""

from __future__ import annotations

import json

import pytest

from tetris.web.build import FRESH, MISSING, STALE, WEB_SRC, build_status, source_hash


@pytest.mark.skipif(not (WEB_SRC / "src").is_dir(), reason="no frontend sources")
def test_committed_build_matches_sources() -> None:
    status = build_status()
    assert status.state == FRESH, f"{status.detail}: run npm run build in web/ and commit it"


def test_status_detects_stale_and_missing_builds(tmp_path) -> None:
    root, static = tmp_path / "web", tmp_path / "static"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.tsx").write_text("one")
    (root / "package.json").write_text("{}")
    assert build_status(root, static).state == MISSING
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>")
    (static / "build-info.json").write_text(json.dumps({"hash": source_hash(root)}))
    assert build_status(root, static).state == FRESH
    (root / "src" / "main.tsx").write_text("two")
    assert build_status(root, static).state == STALE
