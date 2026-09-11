"""mlflow tracking helpers: every VQE run logs its config, environment, and git SHA."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

#: Packages whose exact version matters for reproducing a run's numerics.
_TRACKED_PACKAGES = [
    "qiskit",
    "qiskit-aer",
    "qiskit-nature",
    "qiskit-algorithms",
    "qiskit-ibm-runtime",
    "pennylane",
    "pennylane-lightning",
    "pennylane-qiskit",
    "pyscf",
    "openfermion",
    "mitiq",
    "mthree",
    "numpy",
    "scipy",
]


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _git_is_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout.strip())


def capture_environment() -> dict[str, Any]:
    """Snapshot everything needed to reproduce this run's numerics later."""
    package_versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            package_versions[pkg] = version(pkg)
        except PackageNotFoundError:
            package_versions[pkg] = None

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": package_versions,
        "git_commit_sha": _git_commit_sha(),
        "git_dirty": _git_is_dirty(),
    }


def _flatten_for_mlflow(cfg: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """mlflow params must be flat scalars; flatten nested config dicts to dotted keys."""
    items: dict[str, Any] = {}
    for key, value in cfg.items():
        full_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(_flatten_for_mlflow(value, full_key))
        else:
            items[full_key] = value
    return items


@contextmanager
def tracked_run(
    cfg: DictConfig, experiment_name: str, tracking_uri: str = "file:./mlruns"
) -> Iterator[None]:
    """Start an mlflow run that logs the resolved Hydra config, env snapshot, and git SHA.

    Usage:
        with tracked_run(cfg, experiment_name="vqe-h2"):
            mlflow.log_metric("energy", e)
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(_flatten_for_mlflow(OmegaConf.to_container(cfg, resolve=True)))  # type: ignore[arg-type]

        env = capture_environment()
        if env["git_commit_sha"] is not None:
            mlflow.set_tag("git_commit_sha", env["git_commit_sha"])
            mlflow.set_tag("git_dirty", str(env["git_dirty"]))

        env_path = Path("environment_snapshot.json")
        env_path.write_text(json.dumps(env, indent=2))
        mlflow.log_artifact(str(env_path))
        env_path.unlink()

        yield
