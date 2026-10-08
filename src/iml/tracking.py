"""Thin MLflow wrapper. Active only when ``MLFLOW_TRACKING_URI`` is set (or a uri is passed);
otherwise every call is a no-op and the run folder remains the single source of truth.

Server: ``make mlflow-ui`` starts MLflow on sqlite; locally ``ssh -L 5000:localhost:5000``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from iml.config import ExperimentConfig


class Tracker:
    def __init__(self, uri: str | None = None, enabled: bool | None = None) -> None:
        self.uri = uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.enabled = bool(self.uri) if enabled is None else enabled
        self._mlflow = None
        if self.enabled:
            import mlflow

            mlflow.set_tracking_uri(self.uri)
            self._mlflow = mlflow

    def start(self, cfg: ExperimentConfig, run_name: str) -> None:
        if not self._mlflow:
            return
        self._mlflow.set_experiment(f"{cfg.member}/{cfg.topic}")
        self._mlflow.start_run(
            run_name=run_name,
            tags={"member": cfg.member, "model": cfg.model, **{t: "1" for t in cfg.tags}},
        )
        self._mlflow.log_params(
            {k: v for k, v in cfg.to_dict().items() if not isinstance(v, dict | list)}
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._mlflow:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}, step=step
            )

    def log_artifacts(self, directory: str | Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifacts(str(directory))

    def end(self) -> None:
        if self._mlflow:
            self._mlflow.end_run()
