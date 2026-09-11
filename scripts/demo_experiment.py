"""Phase 0.6 proof: a Hydra-configured, seeded, mlflow-tracked run end to end.

This is scaffolding, not a physics experiment -- it logs a synthetic metric to
prove config composition + seeding + tracking work together. Real VQE runs
(Phase 1+) will use the same `tracked_run` / `seed_everything` primitives.

Usage:
    uv run python scripts/demo_experiment.py
    uv run python scripts/demo_experiment.py seed=123 mlflow.experiment_name=foo
"""

from __future__ import annotations

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from vqe_nisq_project.experiment import seed_everything, tracked_run


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    # Stand-in for a real VQE energy estimate -- proves seeding determinism.
    synthetic_energy = float(np.random.default_rng(cfg.seed).normal(-1.137, 0.01))

    with tracked_run(
        cfg, experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.tracking_uri
    ):
        mlflow.log_metric("synthetic_energy_ha", synthetic_energy)
        print(f"seed={cfg.seed} -> synthetic_energy_ha={synthetic_energy:.10f}")


if __name__ == "__main__":
    main()
