"""Verify environment capture and mlflow run logging actually record what we claim they do."""

from pathlib import Path

import mlflow
from omegaconf import OmegaConf

from vqe_nisq_project.experiment.tracking import _TRACKED_PACKAGES, capture_environment, tracked_run


def test_capture_environment_has_expected_shape() -> None:
    env = capture_environment()

    assert isinstance(env["python_version"], str)
    assert isinstance(env["platform"], str)
    assert set(env["package_versions"]) == set(_TRACKED_PACKAGES)
    # Every tracked package is a real, installed project dependency -- none should be missing.
    assert all(v is not None for v in env["package_versions"].values())


def test_capture_environment_reports_git_sha_in_this_repo() -> None:
    env = capture_environment()
    sha = env["git_commit_sha"]
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_tracked_run_logs_params_metric_and_artifact(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'test_mlflow.db'}"
    cfg = OmegaConf.create({"seed": 7, "note": "pytest"})

    with tracked_run(cfg, experiment_name="pytest-experiment", tracking_uri=tracking_uri):
        mlflow.log_metric("dummy_metric", 1.5)
        run_id = mlflow.active_run().info.run_id

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["seed"] == "7"
    assert run.data.params["note"] == "pytest"
    assert run.data.metrics["dummy_metric"] == 1.5
    assert run.data.tags["git_commit_sha"] is not None

    artifact_names = [a.path for a in client.list_artifacts(run_id)]
    assert "environment_snapshot.json" in artifact_names
