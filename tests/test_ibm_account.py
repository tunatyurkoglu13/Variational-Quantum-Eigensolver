"""No real IBM account/token needed here -- just verify the .env-reading contract."""

import pytest

from vqe_nisq_project.ibm_account import load_ibm_api_token


def test_returns_none_when_token_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IBM_QUANTUM_API_TOKEN", raising=False)
    monkeypatch.chdir("/tmp")  # avoid picking up this repo's real .env, if any
    assert load_ibm_api_token() is None


def test_reads_token_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBM_QUANTUM_API_TOKEN", "dummy-test-token")
    assert load_ibm_api_token() == "dummy-test-token"
