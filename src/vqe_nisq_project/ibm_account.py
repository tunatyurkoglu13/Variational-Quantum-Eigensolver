"""Loads the IBM Quantum API token from .env -- never hardcode it, never commit it."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def load_ibm_api_token() -> str | None:
    """Read IBM_QUANTUM_API_TOKEN from the environment, loading .env first if present."""
    load_dotenv()
    return os.environ.get("IBM_QUANTUM_API_TOKEN") or None
