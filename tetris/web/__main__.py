"""Start the web UI server and open the browser.

Usage:
    python -m tetris.web [--port 8765] [--no-browser] [--checkpoint-dir DIR]
                         [--demo] [--demo-config configs/demo.yaml]

Works offline: the frontend build is committed in tetris/web/static and every
font is self-hosted. If web/src has changed since that build, the build is
redone when Node is installed, and a warning is printed when it is not.
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

from tetris.agents.registry import CHECKPOINT_ENV
from tetris.web import build


def _open_when_up(host: str, port: int, url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.1)


def _config(demo, state_dir: str | None):
    from tetris.web.app import WebConfig

    if state_dir is None:
        return WebConfig(demo=demo)
    root = Path(state_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return WebConfig(
        demo=demo,
        settings_path=root / "user_settings.yaml",
        history_path=root / "user_history.json",
        recordings_dir=root / "recordings",
    )


def _check_build() -> None:
    status = build.build_status()
    if status.ok:
        return
    print(f"Frontend: {status.detail}. Rebuilding...", flush=True)
    if build.rebuild() and build.build_status().ok:
        return
    line = "=" * 72
    print(
        f"{line}\nWARNING: {build.build_status().detail}, and it could not be rebuilt "
        "(needs Node and npm install in web/).\nThe page may be out of date or missing. "
        f"Run: cd web && npm install && npm run build\n{line}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=f"read-only checkpoint directory searched first (sets {CHECKPOINT_ENV})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="preselect the demo opponents and seed, start on the demo look, "
        "F9 skips to results, F10 cycles looks",
    )
    parser.add_argument(
        "--demo-config", default=None, help="demo settings (default configs/demo.yaml)"
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="keep settings and match history here instead of configs/ (rehearsals, tests)",
    )
    args = parser.parse_args()
    _check_build()
    if args.checkpoint_dir:
        os.environ[CHECKPOINT_ENV] = str(Path(args.checkpoint_dir).expanduser())

    import uvicorn

    from tetris.web.app import create_app
    from tetris.web.demo import DEFAULT_DEMO_PATH, DemoConfig, DemoConfigError

    demo = None
    if args.demo:
        try:
            demo = DemoConfig.load(
                Path(args.demo_config) if args.demo_config else DEFAULT_DEMO_PATH
            )
        except (OSError, DemoConfigError) as exc:
            raise SystemExit(f"Demo mode: {exc}") from exc
        shown = ", ".join(f"{a} {p}" for a, p in demo.opponents)
        print(f"Demo mode: {shown}; seed {demo.seed}; look {demo.look}", flush=True)

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Thread(
            target=_open_when_up, args=(args.host, args.port, url), daemon=True
        ).start()
    print(f"Tetris ML Arena web UI at {url}", flush=True)
    uvicorn.run(
        create_app(_config(demo, args.state_dir)),
        host=args.host,
        port=args.port,
        log_level="warning",
        ws="websockets-sansio",
    )


if __name__ == "__main__":
    main()
