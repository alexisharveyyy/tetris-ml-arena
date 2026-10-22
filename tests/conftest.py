"""Every test run keeps its state in a temporary directory and fails if the
real user state files change (:mod:`tests.state_guard`). Imported first, before
any tetris module.

No test may send a request to a real model backend (a local Ollama or a paid
API): the guard below fails any test that reaches the network through the LLM
backends. Tests that exercise the HTTP path patch ``urlopen`` themselves."""

from tests.state_guard import pytest_sessionfinish, pytest_sessionstart  # noqa: I001 (first)

import pytest  # noqa: E402

from tetris.agents.llm import backends as llm_backends  # noqa: E402

__all__ = ["pytest_sessionfinish", "pytest_sessionstart"]


@pytest.fixture(autouse=True)
def no_real_model_requests(monkeypatch):
    sent: list[str] = []

    def refuse(req, timeout=None):
        sent.append(getattr(req, "full_url", str(req)))
        raise llm_backends.BackendError("tests may not call a real model backend")

    monkeypatch.setattr(llm_backends.urllib.request, "urlopen", refuse)
    yield
    assert not sent, f"test sent real model requests: {sent[:3]}"
