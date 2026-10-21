"""WebSocket endpoint: ``/ws?match=<id>`` (see :mod:`tetris.web.protocol`)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from tetris.web.protocol import PROTOCOL_VERSION, Hello, Input, Pause, Ping, client_message, dumps

router = APIRouter()

#: Close code when the match id is unknown (for example after a server restart).
CLOSE_NO_MATCH = 4404


@router.websocket("/ws")
async def match_socket(websocket: WebSocket, match: str) -> None:
    session = websocket.app.state.arena.sessions.get(match)
    await websocket.accept()
    if session is None or session.closed:
        await websocket.send_text(dumps({"t": "error", "message": f"no match {match}"}))
        await websocket.close(code=CLOSE_NO_MATCH)
        return

    lock = asyncio.Lock()  # the match loop and this handler both send

    async def send(text: str) -> None:
        async with lock:
            await websocket.send_text(text)

    m = session.match
    await send(
        dumps(
            {
                "t": "welcome",
                "protocol": PROTOCOL_VERSION,
                "match": session.id,
                "mode": m.config.mode,
                "seed": m.config.seed,
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
        )
    )
    token = session.attach(send)
    try:
        while True:
            text = await websocket.receive_text()
            try:
                msg = client_message.validate_json(text)
            except ValidationError as exc:
                await send(dumps({"t": "error", "message": f"bad message: {exc.errors()[0]}"}))
                continue
            if isinstance(msg, Input):
                session.handle_input(msg.action, msg.down, msg.seq, msg.ts)
            elif isinstance(msg, Pause):
                session.set_paused(msg.paused)
            elif isinstance(msg, Ping):
                await send(dumps({"t": "pong", "ts": msg.ts}))
            elif isinstance(msg, Hello):
                pass  # resume needs nothing extra: the next snapshot is complete
    except WebSocketDisconnect:
        pass
    finally:
        session.detach(token)
