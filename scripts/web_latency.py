"""Measure the web protocol on a running server: message size and input-to-snapshot latency.

Starts a Marathon over the REST API, connects a WebSocket client, taps left and
right at a steady rate and records, for each key press, the time until the
first snapshot that acknowledges it arrives. This is the server half of
input-to-render latency; the browser adds its draw on the next animation frame.

Usage:
    python -m tetris.web --no-browser &         # in another shell
    python scripts/web_latency.py [--url http://127.0.0.1:8765] [--seconds 15]
        [--agents ga cem dqn] [--preset Unlimited]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.request

import websockets


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


async def run(args: argparse.Namespace) -> None:
    info = post(
        f"{args.url}/api/matches",
        {
            "mode": "marathon",
            "opponents": [{"agent": a, "preset": args.preset} for a in args.agents],
        },
    )
    ws_url = args.url.replace("http", "ws", 1) + info["ws"]
    sizes: list[int] = []
    latencies: list[float] = []
    sent: dict[int, float] = {}
    snaps = 0
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"t": "hello", "last_frame": None}))

        async def reader() -> None:
            nonlocal snaps
            async for text in ws:
                now = time.perf_counter()
                msg = json.loads(text)
                if msg["t"] != "snap":
                    continue
                snaps += 1
                sizes.append(len(text))
                ack = msg["ack"]
                if ack is not None:
                    for seq in [s for s in sent if s <= ack["seq"]]:
                        latencies.append((now - sent.pop(seq)) * 1000)

        task = asyncio.create_task(reader())
        # Wait for agents to load and the countdown to finish.
        await asyncio.sleep(args.warmup)
        seq = 0
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            for action in ("left", "right"):
                seq += 1
                sent[seq] = time.perf_counter()
                await ws.send(
                    json.dumps({"t": "input", "seq": seq, "action": action, "down": True, "ts": 0})
                )
                await asyncio.sleep(0.03)
                seq += 1
                await ws.send(
                    json.dumps({"t": "input", "seq": seq, "action": action, "down": False, "ts": 0})
                )
                sent.pop(seq, None)
                await asyncio.sleep(0.07)
        task.cancel()
    elapsed = args.seconds + args.warmup
    print(f"agents: {', '.join(args.agents)} at {args.preset}")
    print(f"snapshots: {snaps} in {elapsed:.0f} s ({snaps / elapsed:.1f}/s)")
    print(
        f"snapshot size: mean {statistics.mean(sizes):.0f} B, p99 {pct(sizes, 0.99)} B, "
        f"max {max(sizes)} B, {statistics.mean(sizes) * snaps / elapsed / 1024:.0f} KiB/s"
    )
    print(
        f"input-to-snapshot latency (n={len(latencies)}): p50 {pct(latencies, 0.5):.1f} ms, "
        f"p99 {pct(latencies, 0.99):.1f} ms, max {max(latencies):.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup", type=float, default=6.0)
    parser.add_argument("--agents", nargs="+", default=["ga", "cem", "dellacherie"])
    parser.add_argument("--preset", default="Unlimited")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
