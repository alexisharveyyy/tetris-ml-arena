"""REST endpoints under ``/api``."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import Counter
from dataclasses import replace
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from tetris.agents.checkpoint_check import CHECKING
from tetris.agents.difficulty import PRESETS
from tetris.agents.identity import GROUPS, HUMAN, IDENTITIES, LOOKS, Identity, identity
from tetris.agents.registry import AGENTS
from tetris.match.controller import MatchConfig, PlayerSpec
from tetris.match.lobby import HUMAN_ID, MAX_OPPONENTS, Lobby, OpponentSlot
from tetris.match.recording import RECORD_SETTINGS, recording_totals
from tetris.match.rules import RaceRules
from tetris.settings import ACTIONS, Settings
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
#: How long starting a match waits for an opponent's load check to finish.
CHECK_WAIT_S = 20.0


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


def _turned_off(spec: Any) -> bool:
    """Opt-in agents (the LLM agent) are off unless their config enables them."""
    if not getattr(spec, "opt_in", False):
        return False
    return not (getattr(spec, "agent_config", None) or {}).get("enabled", False)


def agent_entries(state: AppState) -> list[dict[str, Any]]:
    history = state.history()
    entries = []
    for ident in lineup_identities():
        spec = AGENTS.get(ident.key)
        # ``planned`` only describes identities without a registry spec: once an
        # agent is registered (Phase 6 adds dt and llm), its checks decide.
        if spec is None:
            status = {"state": "planned", "detail": "not built yet"}
            available = False
        elif _turned_off(spec):
            status = {"state": "disabled", "detail": "turned off in its config"}
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


@router.get("/settings/defaults")
def get_settings_defaults() -> dict[str, Any]:
    return Settings().to_dict()


@router.put("/settings")
async def put_settings(patch: SettingsPatch, request: Request) -> dict[str, Any]:
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
    """Playable agents a lobby may keep in its slots.

    Agents whose startup load check is still running count, so a saved lobby is
    not emptied just because the page was opened before the checks finished.
    """
    return [
        e["key"]
        for e in agent_entries(state)
        if e["playable"] and (e["available"] or e["status"]["state"] == CHECKING)
    ]


@router.get("/lobby/{mode}")
def get_lobby(mode: Literal["marathon", "race"], request: Request) -> dict[str, Any]:
    state = _state(request)
    available = _available_playable(state)
    saved = state.settings().last_match.get(mode)
    demo = state.config.demo
    if demo is not None:
        saved = state.demo_lobbies.get(mode) or {**(saved or {}), "slots": demo.lobby_slots()}
    lobby = Lobby.from_dict(mode, saved, available) if saved else Lobby.default(mode, available)
    return {"mode": mode, **lobby.to_dict()}


@router.get("/demo")
def get_demo(request: Request) -> dict[str, Any]:
    demo = _state(request).config.demo
    return {"enabled": False} if demo is None else {"enabled": True, **demo.to_dict()}


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
    mode: Literal["marathon", "race", "watch", "solo", "record"]
    opponents: list[OpponentIn] = []
    seed: int | None = Field(None, ge=0)
    race: dict[str, Any] | None = None
    #: Record mode: "plain" (level gravity) or "garbage" (the training setting).
    record_setting: Literal["plain", "garbage"] = "plain"


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
    rec = session.recorder
    return {
        "match": session.id,
        "mode": "record" if session.record_setting else m.config.mode,
        "recording": None
        if rec is None
        else {
            "setting": session.record_setting,
            "on_space": rec.on_space,
            "off_space": rec.off_space,
            "saved": session.saved_recording,
        },
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
        "record": (0, 0),
    }
    lo, hi = limits[body.mode]
    if not lo <= count <= hi:
        raise HTTPException(422, f"{body.mode} takes {lo} to {hi} opponents, got {count}")
    entries = {e["key"]: e for e in agent_entries(state)}
    wanted = {o.agent for o in body.opponents}
    deadline = time.perf_counter() + CHECK_WAIT_S
    while (
        any(entries.get(k, {}).get("status", {}).get("state") == CHECKING for k in wanted)
        and time.perf_counter() < deadline
    ):
        await asyncio.sleep(0.1)
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
        if state.config.demo is not None:
            state.demo_lobbies[body.mode] = lobby.to_dict()
        else:
            settings.last_match[body.mode] = lobby.to_dict()
            state.save_settings(settings)
        race = lobby.race if body.mode == "race" else None

    await state.close_sessions()  # one match at a time: agents hold worker processes
    seed = body.seed
    if seed is None and state.config.demo is not None and body.mode in LOBBY_MODES:
        seed = state.config.demo.seed
    if seed is None:
        seed = secrets.randbelow(100_000)
    record = body.mode == "record"
    config = MatchConfig(
        mode="solo" if record else body.mode,
        seed=seed,
        race=race,
        eval_garbage_every=(RECORD_SETTINGS[body.record_setting].garbage_every if record else None),
    )
    specs = _player_specs(body.mode, body.opponents)
    runner = state.config.runner_factory() if body.opponents else None
    session = MatchSession(
        config,
        specs,
        runner,
        settings.repeat_settings(),
        history=state.history() if body.mode == "marathon" else None,
        countdown=state.config.start_countdown if body.mode in LOBBY_MODES else 0.0,
        record_setting=body.record_setting if record else None,
    )
    state.sessions[session.id] = session
    session.start()
    return _session_info(session)


# Endpoints that read or change a live session are ``async def`` so they run on
# the event loop with the session's 60 Hz loop. A plain ``def`` endpoint runs in
# a worker thread, where it could end a match while the loop is stepping it (and
# record it in the history twice).


def _session(request: Request, match_id: str) -> MatchSession:
    session = _state(request).sessions.get(match_id)
    if session is None:
        raise HTTPException(404, f"no match {match_id}")
    return session


@router.get("/matches/{match_id}")
async def get_match(match_id: str, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    m = session.match
    return {
        **_session_info(session),
        "standings": m.standings if m.finished else None,
        "history_id": session.history_id,
    }


@router.post("/matches/{match_id}/end")
async def end_match(match_id: str, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    session.end()
    return await get_match(match_id, request)


class RematchIn(BaseModel):
    new_seed: bool = False
    seed: int | None = Field(None, ge=0)


@router.post("/matches/{match_id}/rematch")
async def rematch(match_id: str, body: RematchIn, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    seed = body.seed
    if seed is None and body.new_seed:
        seed = secrets.randbelow(100_000)
    session.rematch(seed)
    return _session_info(session)


@router.post("/matches/{match_id}/keep_watching")
async def keep_watching(match_id: str, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    if not session.match.finished:
        raise HTTPException(409, "the match is still running")
    session.keep_watching()
    return _session_info(session)


class PauseIn(BaseModel):
    paused: bool


@router.post("/matches/{match_id}/pause")
async def pause(match_id: str, body: PauseIn, request: Request) -> dict[str, Any]:
    session = _session(request, match_id)
    session.set_paused(body.paused)
    return {"paused": session.match.paused}


@router.delete("/matches/{match_id}", status_code=204)
async def delete_match(match_id: str, request: Request) -> None:
    state = _state(request)
    session = _session(request, match_id)
    await session.close()
    state.sessions.pop(match_id, None)


@router.get("/recordings")
def get_recordings(request: Request) -> dict[str, Any]:
    root = _state(request).config.recordings_dir
    return {"settings": list(RECORD_SETTINGS), "totals": recording_totals(root)}


@router.post("/matches/{match_id}/recording/save")
async def save_recording(match_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    session = _session(request, match_id)
    if session.recorder is None:
        raise HTTPException(409, "this match is not being recorded")
    if not session.match.finished:
        raise HTTPException(409, "the game is still running")
    if session.saved_recording is None:
        session.save_recording(state.config.recordings_dir)
    return {**_session_info(session), "totals": recording_totals(state.config.recordings_dir)}


class DifficultyIn(BaseModel):
    player: str
    preset: Literal["Easy", "Medium", "Hard", "Unlimited"]


@router.post("/matches/{match_id}/difficulty")
async def set_difficulty(match_id: str, body: DifficultyIn, request: Request) -> dict[str, Any]:
    """Change an agent's speed mid-match (Watch AI)."""
    session = _session(request, match_id)
    player = session.match.players.get(body.player)
    if player is None or player.spec.kind != "agent":
        raise HTTPException(422, f"{body.player} is not an agent player")
    session.match.set_difficulty(body.player, PRESETS[body.preset])
    player.spec.preset = body.preset
    return _session_info(session)
