"""Every test run keeps its state in a temporary directory and fails if the
real user state files change (:mod:`tests.state_guard`). Imported first, before
any tetris module."""

from tests.state_guard import pytest_sessionfinish, pytest_sessionstart

__all__ = ["pytest_sessionfinish", "pytest_sessionstart"]
