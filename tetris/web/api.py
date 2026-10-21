"""REST endpoints under ``/api``."""

from __future__ import annotations

import secrets
from collections import Counter
from dataclasses import replace
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from tetris.agents.difficulty import PRESETS
from tetris.agents.identity import GROUPS, HUMAN, IDENTITIES, LOOKS, Identity, identity
from tetris.agents.registry import AGENTS
from tetris.match.controller import MatchConfig, PlayerSpec
from tetris.match.lobby import HUMAN_ID, MAX_OPPONENTS, Lobby, OpponentSlot
from tetris.match.rules import RaceRules
from tetris.settings import ACTIONS
from tetris.web import leaderboard
from tetris.web.app import AppState
from tetris.web.protocol import json_schema
from tetris.web.session import MatchSession

router = APIRouter(prefix="/api")

#: One line per difficulty for the opponent screen.
PRESET_TEXT: dict[str, str] = {
    "Easy": "About one piece every two seconds, with frequent mistakes.",
    "Medium": "About one piece per second.",
    "Hard": "About two pieces per second, rare mistakes.",
    "Unlimited": "As fast as the model can decide, with no deliberate mistakes.",
}
LOBBY_MODES = ("marathon", "race")


def _state(request: Request) -> AppState:
    return request.app.state.arena


# ------------------------------------------------------------------ agents


def lineup_identities() -> list[Identity]:
    """Every identity in display order, plus registry agents without one."""
    out = list(IDENTITIES.values())
    for key, spec in AGENTS.items():
        if key not in IDENTITIES:
            group = "main" if spec.playable else "leaderboard"
            out.append(replace(identity(key), name=spec.name, group=group))
    return out


def agent_entries(state: AppState) -> list[dict[str, Any]]:
    history = state.history()
    entries = []
    for ident in lineup_identities():
        spec = AGENTS.get(ident.key)
        if spec is None or ident.planned:
            status = {"state": "planned", "detail": "not built yet"}
            available = False
        else:
            check = state.verifier.status(spec)
            status = {"state": check.state, "detail": check.detail}
            available = check.ok
        records = history.agent_records(ident.key)
        entries.append(
            {
                **ident.to_dict(),
                "registered": spec is not None,
                "available": available,
                "status": status,
                "playable": bool(spec.playable) if spec else ident.group != "leaderboard",
                "on_leaderboard": bool(spec.leaderboard) if spec else True,
                "description": spec.description if spec else ident.blurb,
                "supports_target_return": bool(spec and spec.supports_target_return),
                "record": {
                    p: {**rec, "text": history.describe(ident.key, p)}
                    for p in PRESETS
                    for rec in [records.get(p, {"wins": 0, "losses": 0, "ties": 0})]
                },
            }
        )
    return entries


@router.get("/agents")
def get_agents(request: Request) -> dict[str, Any]:
    return {
        "agents": agent_entries(_state(request)),
        "human": HUMAN.to_dict(),
        "groups": list(GROUPS),
        "looks": list(LOOKS),
    }


@router.get("/presets")
def get_presets() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": PRESET_TEXT.get(name, ""), "params": params.to_dict()}
        for name, params in PRESETS.items()
    ]


# ---------------------------------------------------------------- settings


class SettingsPatch(BaseModel):
    look: Literal[LOOKS] | None = None  # type: ignore[valid-type]
    show_thinking: bool | None = None
    show_ghost: bool | None = None
    das_ms: int | None = Field(None, ge=0, le=1000)
    arr_ms: int | None = Field(None, ge=0, le=500)
    soft_drop_rate: float | None = Field(None, gt=0, le=1000)
    bindings: dict[str, list[str]] | None = None


@router.get("/settings")
def get_settings(request: Request) -> dict[str, Any]:
    return _state(request).settings().to_dict()


@router.put("/settings")
def put_settings(patch: SettingsPatch, request: Request) -> dict[str, Any]:
    state = _state(request)
    settings = state.settings()
    changes = patch.model_dump(exclude_none=True)
    bindings = changes.pop("bindings", None)
    if bindings is not None:
        unknown = set(bindings) - set(ACTIONS)
        if unknown:
            raise HTTPException(422, f"unknown actions: {sorted(unknown)}")
        settings.bindings.update({a: list(keys) for a, keys in bindings.items()})
    for k, v in changes.items():
        setattr(settings, k, v)
    state.save_settings(settings)
    return settings.to_dict()


# ------------------------------------------------------------------- lobby


def _available_playable(state: AppState) -> list[str]:
    return [e["key"] for e in agent_entries(state) if e["available"] and e["playable"]]


@router.get("/lobby/{mode}")
def get_lobby(mode: Literal["marathon", "race"], request: Request) -> dict[str, Any]:
    state = _state(request)
    available = _available_playable(state)
    saved = state.settings().last_match.get(mode)
    lobby = Lobby.from_dict(mode, saved, available) if saved else Lobby.default(mode, available)
    return {"mode": mode, **lobby.to_dict()}


# ------------------------------------------------------------- leaderboard


@router.get("/leaderboard")
def get_leaderboard(request: Request) -> dict[str, Any]:
    state = _state(request)
    entries = [e for e in agent_entries(state) if e["on_leaderboard"]]
    keys = [e["key"] for e in entries]
    name_to_key = {spec.name: key for key, spec in AGENTS.items()}
    reports = state.config.reports_dir
    return {
        "survival": leaderboard.survival(reports, keys, name_to_key),
        "marathon": leaderboard.marathon(reports, keys),
        "records": {e["key"]: e["record"] for e in entries},
    }


@router.get("/history")
def get_history(request: Request, limit: int = 50) -> dict[str, Any]:
    history = _state(request).history()
    return {"records": history.records, "matches": history.matches[-limit:][::-1]}


@router.get("/protocol")
def get_protocol() -> dict[str, Any]:
    return json_schema()


# ----------------------------------------------------------------- matches


class OpponentIn(BaseModel):
    agent: str
    preset: Literal["Easy", "Medium", "Hard", "Unlimited"] = "Medium"


class MatchIn(BaseModel):
    mode: Literal["marathon", "race", "watch", "solo"]
    opponents: list[OpponentIn] = []
    seed: int | None = Field(None, ge=0)
    race: dict[str, Any] | None = None


def _player_specs(mode: str, opponents: list[OpponentIn]) -> list[PlayerSpec]:
    specs = [] if mode == "watch" else [PlayerSpec(HUMAN_ID, HUMAN.name)]
    totals = Counter(o.agent for o in opponents)
    seen: Counter[str] = Counter()
    for i, o in enumerate(opponents):
        seen[o.agent] += 1
        name = identity(o.agent).name
        if o.agent not in IDENTITIES:
            name = AGENTS[o.agent].name
        if totals[o.agent] > 1:
            name = f"{name} {seen[o.agent]}"
        specs.append(
            PlayerSpec(f"ai{i + 1}", name, "agent", o.agent, PRESETS[o.preset], preset=o.preset)
        )
    return specs


def _session_info(session: MatchSession) -> dict[str, Any]:
    m = session.match
    return {
        "match": session.id,
        "mode": m.config.mode,
        "seed": m.config.seed,
        "ws": f"/ws?match={session.id}",
        "finished": m.finished,
        "connected": session.has_client,
        "players": [
            {
                "id": s.player_id,
                "kind": s.kind,
                "name": s.name,
                "agent": s.agent_key,
                "preset": s.preset,
            }
            for s in m.specs
        ],
    }


@router.post("/matches", status_code=201)
async def create_match(body: MatchIn, request: Request) -> dict[str, Any]:
    state = _state(request)
    count = len(body.opponents)
    limits = {
        "marathon": (1, MAX_OPPONENTS),
        "race": (1, MAX_OPPONENTS),
        "watch": (1, 1),
        "solo": (0, 0),
    }
    lo, hi = limits[body.mode]
    if not lo <= count <= hi:
        raise HTTPException(422, f"{body.mode} takes {lo} to {hi} opponents, got {count}")
    entries = {e["key"]: e for e in agent_entries(state)}
    for o in body.opponents:
        e = entries.get(o.agent)
        if e is None or not e["registered"]:
            raise HTTPException(422, f"unknown agent {o.agent}")
        if not e["playable"]:
            raise HTTPException(422, f"{o.agent} is not offered as an opponent")
        if not e["available"]:
            raise HTTPException(409, f"{o.agent} is not available: {e['status']['detail']}")

    settings = state.settings()
    race = None
    if body.mode in LOBBY_MODES:
        lobby = Lobby(body.mode)
        saved = settings.last_match.get(body.mode) or {}
        if body.race is not None:
            try:
                lobby.race = RaceRules.from_dict({**RaceRules().to_dict(), **body.race})
            except TypeError as exc:
                raise HTTPException(422, f"bad race rules: {exc}") from exc
        elif saved.get("race"):
            lobby.race = RaceRules.from_dict(saved["race"])
        lobby.slots = [OpponentSlot(o.agent, o.preset) for o in body.opponents] + [None] * (
            MAX_OPPONENTS - count
        )
        settings.last_match[body.mode] = lobby.to_dict()
        state.save_settings(settings)
        race = lobby.race if body.mode == "race" else None

    await state.close_sessions()  # one match at a time: agents hold worker processes
    seed = body.seed if body.seed is not None else secrets.randbelow(100_000)
    config = MatchConfig(mode=body.mode, seed=seed, race=race)
    specs = _player_specs(body.mode, body.opponents)
    runner = state.config.runner_factory() if body.opponents else None
    session = MatchSession(
        config,
        specs,
        runner,
        settings.repeat_settings(),
        history=state.history() if body.mode == "marathon" else None,
        countdown=state.config.start_countdown if body.mode in LOBBY_MODES else 0.0,
    )
    state.sessions[session.id] = session
    session.start()
    return _session_info(session)


def _session(request: Request, match_id: str) -> MatchSession:
    session = _state(request).sessions.get(match_id)
    if session is None:
        raise HTTPException(404, f"no match {match_id}")
    return session


@router.get("/matches/{match_id}")
def get_match(match_id: str, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    m = session.match
    return {
        **_session_info(session),
        "standings": m.standings if m.finished else None,
        "history_id": session.history_id,
    }


@router.post("/matches/{match_id}/end")
def end_match(match_id: str, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    session.end()
    return get_match(match_id, request)


class RematchIn(BaseModel):
    new_seed: bool = False
    seed: int | None = Field(None, ge=0)


@router.post("/matches/{match_id}/rematch")
def rematch(match_id: str, body: RematchIn, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    seed = body.seed
    if seed is None and body.new_seed:
        seed = secrets.randbelow(100_000)
    session.rematch(seed)
    return _session_info(session)


@router.post("/matches/{match_id}/keep_watching")
def keep_watching(match_id: str, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    if not session.match.finished:
        raise HTTPException(409, "the match is still running")
    session.keep_watching()
    return _session_info(session)


class PauseIn(BaseModel):
    paused: bool


@router.post("/matches/{match_id}/pause")
def pause(match_id: str, body: PauseIn, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    session.set_paused(body.paused)
    return {"paused": session.match.paused}


@router.delete("/matches/{match_id}", status_code=204)
async def delete_match(match_id: str, request: Request) -> None:
    state = _state(request)
    session = _session(request, match_id)
    await session.close()
    state.sessions.pop(match_id, None)
