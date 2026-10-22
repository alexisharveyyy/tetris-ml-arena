"""LLM backends behind one interface: local Ollama, the OpenAI API, the Anthropic API.

Every backend sends one system prompt and one user prompt, asks for a short
JSON answer at temperature 0 and returns the text with its token usage.
Network failures raise :class:`BackendError`; running past the timeout raises
:class:`BackendTimeout`. Only the standard library is used (``urllib``).

API backends read their key from an environment variable and charge every
call to the :class:`~tetris.agents.llm.spending.SpendingLedger` first, so a
call that could cross the spending cap is never sent. A call the API rejects
with an HTTP error response is settled at $0 (rejected requests are not
billed); the worst-case reservation is kept only when no reply arrives, or a
reply arrives without readable token usage.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from tetris.agents.llm.prompt import RESPONSE_SCHEMA
from tetris.agents.llm.spending import Price, SpendingLedger, estimate_tokens


class BackendError(RuntimeError):
    """The backend could not produce an answer (connection, HTTP or format error).

    ``status`` is the HTTP status code when the server answered with an error
    response (the request was rejected), else None.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class BackendTimeout(BackendError):
    """No answer within the timeout."""


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cost_usd: float = 0.0


class Backend(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, timeout: float) -> Reply: ...


def _post(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except TimeoutError as exc:
        raise BackendTimeout(f"no reply within {timeout:.1f}s") from exc
    except urllib.error.HTTPError as exc:
        body = exc.read()[:300].decode(errors="replace")
        raise BackendError(f"HTTP {exc.code}: {body}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError | socket.timeout):
            raise BackendTimeout(f"no reply within {timeout:.1f}s") from exc
        raise BackendError(f"connection failed: {exc.reason}") from exc
    except ValueError as exc:
        raise BackendError(f"reply is not JSON: {exc}") from exc


class OllamaBackend:
    """A local model served by Ollama (``ollama serve``). Free; runs on this Mac's GPU."""

    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", **opts: Any) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.keep_alive = opts.get("keep_alive", "15m")
        self.max_output_tokens = int(opts.get("max_output_tokens", 24))
        self.thinking_toggle = True

    def complete(self, system: str, user: str, timeout: float) -> Reply:
        t0 = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": self.max_output_tokens},
        }
        if self.thinking_toggle:
            payload["think"] = False  # reasoning models (qwen3): answer directly
        try:
            data = _post(f"{self.host}/api/chat", payload, {}, timeout)
        except BackendError as exc:
            # Models without a thinking mode (llama3.1) may reject the flag: retry without it.
            if not self.thinking_toggle or "think" not in str(exc).lower():
                raise
            self.thinking_toggle = False
            payload.pop("think")
            data = _post(
                f"{self.host}/api/chat", payload, {}, max(0.1, timeout - (time.perf_counter() - t0))
            )
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BackendError(f"unexpected Ollama reply: {str(data)[:200]}") from exc
        return Reply(
            text,
            int(data.get("prompt_eval_count", 0)),
            int(data.get("eval_count", 0)),
            time.perf_counter() - t0,
        )


class _PaidBackend:
    """Shared spending logic for API backends."""

    name = "api"

    def __init__(
        self,
        model: str,
        api_key_env: str,
        price: Price,
        ledger: SpendingLedger,
        max_output_tokens: int = 24,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.price = price
        self.ledger = ledger
        self.max_output_tokens = int(max_output_tokens)

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise BackendError(f"{self.api_key_env} is not set")
        return key

    def _request(self, system: str, user: str, timeout: float) -> tuple[str, int, int]:
        raise NotImplementedError

    def complete(self, system: str, user: str, timeout: float) -> Reply:
        key_check = self.api_key  # fail before reserving if there is no key
        del key_check
        reserve = self.price.cost(estimate_tokens(system + user), self.max_output_tokens)
        self.ledger.reserve(reserve)  # raises BudgetExhausted; nothing sent
        t0 = time.perf_counter()
        tokens = (0, 0)
        # If no reply arrives (or a reply has no readable usage), assume the worst
        # case was billed.
        actual = reserve
        try:
            text, tin, tout = self._request(system, user, timeout)
            tokens = (tin, tout)
            actual = self.price.cost(tin, tout)
        except BackendError as exc:
            if exc.status is not None:  # rejected with an error response: not billed
                actual = 0.0
            raise
        finally:
            self.ledger.settle(reserve, actual, self.model, tokens)
        return Reply(text, tokens[0], tokens[1], time.perf_counter() - t0, actual)


class OpenAIBackend(_PaidBackend):
    name = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def _request(self, system: str, user: str, timeout: float) -> tuple[str, int, int]:
        data = _post(
            self.url,
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_completion_tokens": self.max_output_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "placement", "strict": True, "schema": RESPONSE_SCHEMA},
                },
            },
            {"Authorization": f"Bearer {self.api_key}"},
            timeout,
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data["usage"]
            return text, int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected OpenAI reply: {str(data)[:200]}") from exc


class AnthropicBackend(_PaidBackend):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def _request(self, system: str, user: str, timeout: float) -> tuple[str, int, int]:
        data = _post(
            self.url,
            {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            timeout,
        )
        try:
            text = "".join(b.get("text", "") for b in data["content"])
            usage = data["usage"]
            return text, int(usage["input_tokens"]), int(usage["output_tokens"])
        except (KeyError, TypeError) as exc:
            raise BackendError(f"unexpected Anthropic reply: {str(data)[:200]}") from exc


def make_backend(cfg: dict[str, Any], name: str | None = None) -> Backend:
    """The backend named in ``cfg`` (or ``name``), with prices and the ledger for APIs."""
    from tetris.agents.registry import ROOT

    name = name or cfg["backend"]
    max_out = int(cfg.get("max_output_tokens", 24))
    if name == "ollama":
        o = cfg["ollama"]
        return OllamaBackend(
            o["model"], o["host"], keep_alive=o.get("keep_alive"), max_output_tokens=max_out
        )
    if name not in ("openai", "anthropic"):
        raise ValueError(f"unknown LLM backend {name!r}")
    section = cfg[name]
    model = section["model"]
    if model not in cfg["prices"]:
        raise ValueError(f"no price for {model} in the LLM config; add it before using the API")
    price = Price(**cfg["prices"][model])
    ledger = SpendingLedger(ROOT / cfg["spending"]["ledger"], float(cfg["spending"]["cap_usd"]))
    cls = OpenAIBackend if name == "openai" else AnthropicBackend
    return cls(model, section["api_key_env"], price, ledger, max_out)
