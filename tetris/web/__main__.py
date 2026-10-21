"""Start the web UI server and open the browser.

Usage:
    python -m tetris.web [--port 8765] [--no-browser] [--checkpoint-dir DIR]
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


def _open_when_up(host: str, port: int, url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.1)


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
    args = parser.parse_args()
    if args.checkpoint_dir:
        os.environ[CHECKPOINT_ENV] = str(Path(args.checkpoint_dir).expanduser())

    import uvicorn

    from tetris.web.app import create_app

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Thread(
            target=_open_when_up, args=(args.host, args.port, url), daemon=True
        ).start()
    print(f"Tetris ML Arena web UI at {url}", flush=True)
    uvicorn.run(
        create_app(), host=args.host, port=args.port, log_level="warning", ws="websockets-sansio"
    )


if __name__ == "__main__":
    main()
