"""WebSocket protocol, version 1: message models and the snapshot encoder.

One socket per client, JSON text frames, each with a ``t`` type field.

Client to server: ``hello``, ``input``, ``pause``, ``ping``.
Server to client: ``welcome``, ``snap`` (60 per second), ``pong``, ``error``.

Boards are 200-character strings of the visible 10x20 playfield, row-major from
the top: ``.`` empty, ``IJLOSTZ`` locked piece cells, ``G`` garbage. Cell
coordinates are ``[row, col]`` in the same visible frame; rows below 0 are in
the hidden spawn rows above it.

``python -m tetris.web.protocol`` prints the JSON schema of every message.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tetris.engine.board import HIDDEN_ROWS, VISIBLE_HEIGHT, WIDTH, board_to_strings, piece_cells
from tetris.match.controller import MatchController, PlayerState
from tetris.match.feed import FALLBACK_TEXT, FEED_TYPES

PROTOCOL_VERSION = 1
ACTIONS = ("left", "right", "soft", "hard", "cw", "ccw", "hold")

Cell = tuple[int, int]


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------ client to server


class Hello(_Msg):
    t: Literal["hello"]
    last_frame: int | None = None


class Input(_Msg):
    t: Literal["input"]
    seq: int
    action: Literal["left", "right", "soft", "hard", "cw", "ccw", "hold"]
    down: bool
    ts: float  # client clock, milliseconds; echoed back in ``ack``


class Pause(_Msg):
    t: Literal["pause"]
    paused: bool


class Ping(_Msg):
    t: Literal["ping"]
    ts: float


ClientMessage = Annotated[Hello | Input | Pause | Ping, Field(discriminator="t")]
client_message = TypeAdapter(ClientMessage)


# ------------------------------------------------------------ server to client


class Piece(_Msg):
    p: str
    c: list[Cell]


class Candidate(_Msg):
    rank: int
    hold: bool  # the placement uses hold first
    p: str
    c: list[Cell]
    score: float  # value, or policy probability, depending on the agent


class Stats(_Msg):
    score: int
    lines: int
    level: int
    pps: float
    tetrises: int
    pieces: int


class Status(_Msg):
    state: Literal["playing", "thinking", "topped_out", "done", "fallback", "loading"]
    text: str
    fallback: str | None = None  # reason code while the baseline plays


class ModelMoves(_Msg):
    """Moves of an agent that reports each one (the LLM agent), and how many fell
    back to its heuristic, by cause (timeout, invalid, error, budget, late)."""

    moves: int
    fallbacks: int
    by_cause: dict[str, int]


class Player(_Msg):
    id: str
    kind: Literal["human", "agent"]
    name: str
    agent: str | None
    preset: str | None
    place: int
    board: Annotated[str, Field(min_length=200, max_length=200)]
    active: Piece | None
    ghost: list[Cell] | None
    hold: str | None
    can_hold: bool
    next: list[str]
    stats: Stats
    status: Status
    finish_time: float | None
    cand: list[Candidate]
    #: True when ``cand`` is for the current piece; False when it is the previous
    #: piece's (fast agents place within one frame; see AgentDriver.overlay).
    cand_current: bool
    model: ModelMoves | None = None  # per-move fallbacks (the LLM agent), else None


class FeedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal[FEED_TYPES]  # type: ignore[valid-type]
    player: str | None
    time: float


class Ack(_Msg):
    seq: int
    ts: float


class Race(_Msg):
    condition: str
    time_limit: float
    target_lines: int
    description: str
    time_remaining: float | None


class Snapshot(_Msg):
    t: Literal["snap"]
    f: int
    mode: str
    seed: int
    time: float
    countdown: float
    loading: bool
    paused: bool
    disconnected: bool  # paused because the client connection dropped
    finished: bool
    spectating: bool
    winner: str | None
    standings: list[str]
    final: list[dict[str, Any]] | None
    history_id: str | None
    race: Race | None
    ack: Ack | None
    ev: list[FeedEvent]
    players: list[Player]


class Welcome(_Msg):
    t: Literal["welcome"]
    protocol: int
    match: str
    mode: str
    seed: int
    players: list[dict[str, Any]]


class Pong(_Msg):
    t: Literal["pong"]
    ts: float


class Error(_Msg):
    t: Literal["error"]
    message: str


ServerMessage = Annotated[Welcome | Snapshot | Pong | Error, Field(discriminator="t")]
server_message = TypeAdapter(ServerMessage)


# --------------------------------------------------------------- encoding


def _visible(cells: list[tuple[int, int]]) -> list[Cell]:
    return [(r - HIDDEN_ROWS, c) for r, c in cells]


def encode_board(rows: list[str]) -> str:
    board = "".join(rows[HIDDEN_ROWS:])
    assert len(board) == VISIBLE_HEIGHT * WIDTH
    return board


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _status(match: MatchController, p: PlayerState) -> dict[str, Any]:
    g = p.game
    pid = p.spec.player_id
    if g.game_over:
        when = p.finish_time if p.finish_time is not None else match.time
        return {"state": "topped_out", "text": f"Topped out at {_clock(when)}"}
    if p.done:
        return {"state": "done", "text": "Finished"}
    if p.driver is not None and match.runner is not None:
        reason = match.runner.fallback_reason(pid)
        if reason is not None:
            text = FALLBACK_TEXT.get(reason, FALLBACK_TEXT["crash"])
            return {"state": "fallback", "text": text, "fallback": reason}
        if not match.config.lockstep and not match.ready:
            return {"state": "loading", "text": "Loading the model"}
        if p.driver.thinking:
            return {"state": "thinking", "text": "Thinking"}
    if match.finished and not match.spectating:
        return {"state": "playing", "text": "Still playing"}
    return {"state": "playing", "text": "Playing"}


def encode_player(match: MatchController, p: PlayerState, place: int) -> dict[str, Any]:
    g = p.game
    row = match.player_row(p.spec.player_id)
    active = ghost = None
    a = g.active
    if a is not None and p.alive:
        active = {"p": a.piece.letter, "c": _visible(piece_cells(a.piece, a.rot, a.row, a.col))}
        ghost_row = g.ghost_row()
        if ghost_row is not None:
            ghost = _visible(piece_cells(a.piece, a.rot, ghost_row, a.col))
    cand, current = [], False
    if p.driver is not None and p.alive:
        cells, current = p.driver.overlay(g)
        cand = [
            {
                "rank": c["rank"],
                "hold": c["hold"],
                "p": c["piece"],
                "c": _visible([tuple(x) for x in c["cells"]]),
                "score": float(c["score"]),
            }
            for c in cells
        ]
    return {
        "id": p.spec.player_id,
        "kind": p.spec.kind,
        "name": p.spec.name,
        "agent": p.spec.agent_key,
        "preset": p.spec.preset,
        "place": place,
        "board": encode_board(board_to_strings(g.board)),
        "active": active,
        "ghost": ghost,
        "hold": g.hold.letter if g.hold else None,
        "can_hold": g.can_hold,
        "next": [q.letter for q in g.next_queue],
        "stats": {
            "score": row["score"],
            "lines": row["lines"],
            "level": row["level"],
            "pps": round(row["pps"], 3),
            "tetrises": row["tetrises"],
            "pieces": row["pieces"],
        },
        "status": _status(match, p),
        "finish_time": p.finish_time,
        "cand": cand,
        "cand_current": current,
        "model": _model_moves(row.get("model_moves")),
    }


def _model_moves(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    return {k: summary[k] for k in ("moves", "fallbacks", "by_cause")}


def encode_snapshot(
    match: MatchController,
    frame: int,
    *,
    ack: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    disconnected: bool = False,
    history_id: str | None = None,
) -> dict[str, Any]:
    order = match.live_standings()
    places = {pid: i + 1 for i, pid in enumerate(order)}
    race = None
    rules = match.config.race
    if match.config.mode == "race" and rules is not None:
        race = {
            "condition": rules.condition,
            "time_limit": rules.time_limit,
            "target_lines": rules.target_lines,
            "description": rules.describe(),
            "time_remaining": (
                max(0.0, rules.time_limit - match.time)
                if rules.condition != "first_to_lines"
                else None
            ),
        }
    return {
        "t": "snap",
        "f": frame,
        "mode": match.config.mode,
        "seed": match.config.seed,
        "time": round(match.time, 3),
        "countdown": round(match.countdown, 3),
        "loading": not match.config.lockstep and not match.ready,
        "paused": match.paused,
        "disconnected": disconnected,
        "finished": match.finished,
        "spectating": match.spectating,
        "winner": match.winner,
        "standings": order,
        "final": match.standings if match.finished else None,
        "history_id": history_id,
        "race": race,
        "ack": ack,
        "ev": events or [],
        "players": [encode_player(match, p, places[pid]) for pid, p in match.players.items()],
    }


def dumps(message: dict[str, Any]) -> str:
    """Compact JSON for the wire."""
    return json.dumps(message, separators=(",", ":"))


def json_schema() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "client": client_message.json_schema(),
        "server": server_message.json_schema(),
    }


if __name__ == "__main__":
    print(json.dumps(json_schema(), indent=2))
