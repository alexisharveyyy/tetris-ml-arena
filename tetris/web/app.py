"""FastAPI application: REST API, WebSocket and static pages.

:func:`create_app` takes a :class:`WebConfig` so tests can point every file at a
temporary directory and run agents inline instead of in worker processes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from tetris.agents.checkpoint_check import CheckpointVerifier
from tetris.agents.registry import AGENTS, ROOT
from tetris.match.agent_runtime import AgentRunner, ProcessRunner
from tetris.match.history import DEFAULT_HISTORY_PATH, MarathonHistory
from tetris.settings import DEFAULT_SETTINGS_PATH, Settings
from tetris.web.session import MatchSession

WEB_DIR = Path(__file__).resolve().parent


@dataclass
class WebConfig:
    settings_path: Path = DEFAULT_SETTINGS_PATH
    history_path: Path = DEFAULT_HISTORY_PATH
    reports_dir: Path = ROOT / "reports"
    runner_factory: Callable[[], AgentRunner] = ProcessRunner
    #: Countdown before Marathon and Race matches start.
    start_countdown: float = 3.0
    #: Run checkpoint load tests in-process (tests with light agents only).
    inline_checks: bool = False


@dataclass
class AppState:
    config: WebConfig
    verifier: CheckpointVerifier
    sessions: dict[str, MatchSession] = field(default_factory=dict)

    def settings(self) -> Settings:
        return Settings.load(self.config.settings_path)

    def save_settings(self, settings: Settings) -> None:
        settings.save(self.config.settings_path)
        for session in self.sessions.values():
            session.repeater.settings = settings.repeat_settings()

    def history(self) -> MarathonHistory:
        return MarathonHistory(self.config.history_path)

    async def close_sessions(self) -> None:
        for session in list(self.sessions.values()):
            await session.close()
        self.sessions.clear()


def create_app(config: WebConfig | None = None) -> FastAPI:
    from tetris.web.api import router as api_router
    from tetris.web.ws import router as ws_router

    config = config or WebConfig()
    state = AppState(config, CheckpointVerifier(inline=config.inline_checks))

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        for spec in AGENTS.values():
            state.verifier.status(spec)  # start load checks in the background
        yield
        await state.close_sessions()
        state.verifier.close()

    app = FastAPI(title="Tetris ML Arena", lifespan=lifespan)
    app.state.arena = state
    app.include_router(api_router)
    app.include_router(ws_router)

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse("/debug")

    @app.get("/debug", include_in_schema=False)
    async def debug_page() -> FileResponse:
        return FileResponse(WEB_DIR / "debug" / "index.html")

    return app
