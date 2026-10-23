"""Links and repository paths in tracked files point at files that exist.

Two checks over the files git tracks:

1. Every Markdown link and image that is not a URL resolves relative to the
   file that contains it (anchors are ignored).
2. Every repository path mentioned in a text file (``reports/...``,
   ``scripts/...``, ``docs/...`` and so on) exists, either from the repository
   root or from the file's own folder (``web/`` configs use relative paths).

Records of past runs keep the paths they ran with and are not checked for
(2): the logs, the commit hash map, checkpoint notes, the recorded
leaderboard sources, the commands in the final report's reproduction table
and the old-path table in ``scripts/README.md``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".md", ".py", ".ts", ".tsx", ".yaml", ".yml", ".toml", ".json", ".txt", ".html"}
#: Generated or third-party files: never checked.
SKIP_FILES = ("tetris/web/static/", "web/node_modules/", "web/package-lock.json")
#: Records of past runs: links are checked, mentioned paths are not.
HISTORICAL = (
    "logs/",
    "checkpoints/",
    "docs/commit_hash_map.txt",
    "reports/leaderboard/leaderboard_sources.json",
)
#: Folders that are not in git, so paths under them cannot be checked.
NOT_IN_GIT = (
    "checkpoints/",
    "data/",
    "runs/",
    "logs/dt",
    "logs/ga",
    "web/screenshots",
    "web/node_modules",
    "web/test-results",
    "web/playwright-report",
)
#: Paths a run creates, and examples in docstrings and tests.
NOT_YET_WRITTEN = {
    "reports/leaderboard/leaderboard_rerun_check.md",
    "reports/leaderboard/leaderboard_dt_v1/",
    "scripts/hello.py",
    "tetris/step.py",
}

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REPO_PATH = re.compile(
    r"(?<![\w./~-])((?:reports|docs|scripts|configs|tests|web|tetris|logs)/[\w./-]*[\w/])"
)


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return out.split()


def _text_files(tracked: list[str]) -> dict[str, str]:
    files = {}
    for f in tracked:
        path = ROOT / f
        if path.suffix not in TEXT_SUFFIXES or f.startswith(SKIP_FILES) or not path.is_file():
            continue
        try:
            files[f] = path.read_text()
        except UnicodeDecodeError:
            continue
    return files


@pytest.fixture(scope="module")
def repo() -> tuple[set[str], dict[str, str]]:
    tracked = _tracked()
    known = set(tracked)
    for f in tracked:
        known.update(str(p) for p in Path(f).parents)
    return known, _text_files(tracked)


def _exists(path: str, known: set[str]) -> bool:
    path = path.rstrip("/").rstrip(".")
    return path in known or (ROOT / path).exists()


def _is_record_line(f: str, line: str) -> bool:
    if f == "reports/FINAL_REPORT.md":
        return line.startswith("| ") and "python" in line  # reproduction table commands
    if f == "scripts/README.md":
        return line.startswith("| `scripts/")  # the old-path table
    return False


def test_markdown_links_resolve(repo) -> None:
    _, files = repo
    broken = []
    for f, text in files.items():
        if not f.endswith(".md"):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for m in LINK.finditer(line):
                target = m.group(1).split("#")[0]
                if not target or re.match(r"[a-z]+:", target):
                    continue
                if not ((ROOT / f).parent / target).exists():
                    broken.append(f"{f}:{n}: {m.group(1)}")
    assert not broken, "broken Markdown links:\n" + "\n".join(broken)


def test_mentioned_repo_paths_exist(repo) -> None:
    known, files = repo
    missing = []
    for f, text in files.items():
        if f.startswith(HISTORICAL):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if _is_record_line(f, line):
                continue
            for m in REPO_PATH.finditer(line):
                path = re.sub(r"/\.\w+$", "", m.group(1))  # "name.md/.json" shorthand
                if line[m.end() : m.end() + 1] in ("*", "{", "<") or path.startswith(NOT_IN_GIT):
                    continue  # a glob, a placeholder or a folder outside git
                if path in NOT_YET_WRITTEN:
                    continue
                candidates = (path, str(Path(f).parent / path), "web/" + path)
                if any(_exists(c, known) for c in candidates):
                    continue
                missing.append(f"{f}:{n}: {path}")
    assert not missing, "paths that do not exist:\n" + "\n".join(missing)
