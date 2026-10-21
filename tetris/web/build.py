"""Check that the committed frontend build matches ``web/src``.

The build (``tetris/web/static``) is committed so the demo machine needs no
Node. ``npm run build`` writes ``build-info.json`` with a fingerprint of the
sources (see ``web/scripts/build-info.mjs``); this module computes the same
fingerprint so a stale build is caught by a test and at launch.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tetris.agents.registry import ROOT

WEB_SRC = ROOT / "web"
STATIC = Path(__file__).resolve().parent / "static"
#: Same list, in the same form, as INPUTS in web/scripts/build-info.mjs.
INPUTS = (
    "index.html",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "vite.config.ts",
    "scripts/build-info.mjs",
    "src",
)
SKIP = {".DS_Store"}

FRESH, STALE, MISSING, NO_SOURCE = "fresh", "stale", "missing", "no_source"


def source_hash(root: Path = WEB_SRC) -> str:
    files: list[str] = []
    for item in INPUTS:
        path = root / item
        if path.is_dir():
            files += [
                p.relative_to(root).as_posix()
                for p in path.rglob("*")
                if p.is_file() and not any(part in SKIP for part in p.parts)
            ]
        elif path.is_file():
            files.append(item)
    h = hashlib.sha256()
    for f in sorted(files):
        digest = hashlib.sha256((root / f).read_bytes()).hexdigest()
        h.update(f"{f}\0{digest}\n".encode())
    return h.hexdigest()


@dataclass(frozen=True)
class BuildStatus:
    state: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.state in (FRESH, NO_SOURCE)


def build_status(root: Path = WEB_SRC, static: Path = STATIC) -> BuildStatus:
    info_path = static / "build-info.json"
    if not (static / "index.html").exists() or not info_path.exists():
        return BuildStatus(MISSING, f"no frontend build in {static}")
    if not (root / "src").is_dir():
        return BuildStatus(NO_SOURCE, "no web/src here; using the committed build")
    built = json.loads(info_path.read_text()).get("hash")
    if built != source_hash(root):
        return BuildStatus(STALE, "web/src has changed since the committed build")
    return BuildStatus(FRESH, "build matches web/src")


def rebuild(root: Path = WEB_SRC) -> bool:
    """Run ``npm run build`` if Node and the installed packages are there."""
    npm = shutil.which("npm")
    if npm is None or not (root / "node_modules").is_dir():
        return False
    return subprocess.run([npm, "run", "build"], cwd=root).returncode == 0
