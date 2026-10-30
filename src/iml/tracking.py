"""MLflow tracking for the whole repo: classification-quality metrics only.

Nothing but quality metrics (accuracy, FAR, FRR, EER, F1, precision, recall, ...) goes to MLflow:
no figures, tables or artifacts (the run folder / notebook folder is where those live).

* ``Tracker``: used by ``iml-train``. It is active only when ``MLFLOW_TRACKING_URI`` is set (or a
  uri is passed). Otherwise every call is a no-op and the run folder is the single source of truth.
  Only val/test quality metrics are logged (no loss, lr, timings or counts) and no artifacts.
* ``notebook_run``: one MLflow run per evaluated variant in notebooks and scripts. It uses
  ``MLFLOW_TRACKING_URI`` or ``http://localhost:5000``, and it becomes a no-op (one warning, never
  raises) when the server is unreachable or ``IML_TRACKING=0``. Its artifact methods
  (``log_artifact``/``log_dict``/``log_text``/``log_figure``/``log_table``) are no-ops.

Experiments are named ``<member>/<topic>``. Server: ``make mlflow-up`` (idempotent) or
``make mlflow-ui`` (foreground); locally ``ssh -L 5000:localhost:5000 <server>``.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from iml.config import ExperimentConfig

DEFAULT_URI = "http://localhost:5000"
# MLflow 3's UI opens an untyped experiment on the GenAI "Traces" tab (always empty here)
EXPERIMENT_KIND_TAG, EXPERIMENT_KIND = "mlflow.experimentKind", "custom_model_development"
_HEALTH_TIMEOUT = 1.5
_MAX_KEY = 250  # mlflow.utils.validation.MAX_ENTITY_KEY_LENGTH
_MAX_PARAM = 6000  # MAX_PARAM_VAL_LENGTH
_MAX_TAG = 8000  # MAX_TAG_VAL_LENGTH
_BATCH = 100  # MAX_PARAMS_TAGS_PER_BATCH
_warned: set[str] = set()
# what Tracker.log_metrics keeps: <split>_<metric> or <split>/<metric> of these splits / metrics
QUALITY_SPLITS = ("val", "test")
QUALITY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "f1",
    "far",
    "frr",
    "eer",
    "precision",
    "recall",
)
_QUALITY_KEY = re.compile(rf"^({'|'.join(QUALITY_SPLITS)})[_/]({'|'.join(QUALITY_METRICS)})$")
NO_ARTIFACTS = "artifacts are not logged (quality metrics only)"


def is_quality_metric(key: str) -> bool:
    """``val_far``, ``test/macro_f1``: yes; ``train_loss``, ``lr``, ``val_n``, ``time_s``: no."""
    return bool(_QUALITY_KEY.match(key))


def _no_artifacts() -> None:
    if NO_ARTIFACTS not in _warned:
        _warned.add(NO_ARTIFACTS)
        print(f"[iml.tracking] {NO_ARTIFACTS}", file=sys.stderr)


def tracking_disabled() -> bool:
    return os.environ.get("IML_TRACKING", "").strip().lower() in {"0", "false", "no", "off"}


def server_reachable(uri: str, timeout: float = _HEALTH_TIMEOUT) -> bool:
    """``GET <uri>/health`` for http(s) uris; other uris (sqlite:, file:, paths) count as up."""
    if not uri.startswith(("http://", "https://")):
        return True
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[iml.tracking] WARNING: {msg}", file=sys.stderr)


class Tracker:
    def __init__(self, uri: str | None = None, enabled: bool | None = None) -> None:
        self.uri = uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.enabled = bool(self.uri) if enabled is None else enabled
        if enabled is None and self.enabled and tracking_disabled():
            self.enabled = False
        if enabled is None and self.enabled and not server_reachable(self.uri):
            _warn_once(
                self.uri,
                f"MLflow server {self.uri} is unreachable: training continues without MLflow "
                "(start it with `make mlflow-up`).",
            )
            self.enabled = False
        self._mlflow = None
        if self.enabled:
            import mlflow

            mlflow.set_tracking_uri(self.uri)
            self._mlflow = mlflow

    def start(self, cfg: ExperimentConfig, run_name: str) -> None:
        if not self._mlflow:
            return
        exp = self._mlflow.set_experiment(f"{cfg.member}/{cfg.topic}")
        if EXPERIMENT_KIND_TAG not in exp.tags:
            self._mlflow.set_experiment_tag(EXPERIMENT_KIND_TAG, EXPERIMENT_KIND)
        self._mlflow.start_run(
            run_name=run_name,
            tags={
                "member": cfg.member,
                "model": cfg.model,
                "source": "train",
                **{t: "1" for t in cfg.tags},
            },
        )
        self._mlflow.log_params(
            {k: v for k, v in cfg.to_dict().items() if not isinstance(v, dict | list)}
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Only val/test quality metrics (:func:`is_quality_metric`); the rest is dropped."""
        if self._mlflow:
            keep = {
                k: float(v)
                for k, v in metrics.items()
                if is_quality_metric(k) and isinstance(v, int | float) and not isinstance(v, bool)
            }
            if keep:
                self._mlflow.log_metrics(keep, step=step)

    def log_artifacts(self, directory: str | Path) -> None:
        """No-op: the run folder stays on disk, MLflow gets quality metrics only."""

    def end(self) -> None:
        if self._mlflow:
            self._mlflow.end_run()


# --- notebooks / scripts ----------------------------------------------------------------------


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _key(k: str) -> str:
    # mlflow keys: alphanumerics, _ - . space / (and no leading /)
    return re.sub(r"[^\w\-. /]", "_", str(k)).lstrip("/")[:_MAX_KEY] or "_"


def _str(v: Any, limit: int) -> str:
    if isinstance(v, str):
        s = v
    else:
        s = json.dumps(v, default=str) if isinstance(v, list | tuple) else str(v)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        try:  # numpy 0-d arrays, torch scalars
            if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
                return _number(v.item())
        except Exception:
            pass
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _git(*args: str) -> str | None:
    for cwd in (Path.cwd(), Path(__file__).resolve().parent):
        try:
            r = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
            )
            return r.stdout.strip()
        except Exception:
            continue
    return None


def _source() -> str | None:
    try:
        ip = sys.modules.get("IPython")
        shell = ip.get_ipython() if ip else None
        if shell is not None and shell.user_ns.get("__vsc_ipynb_file__"):
            return str(shell.user_ns["__vsc_ipynb_file__"])
    except Exception:
        pass
    if os.environ.get("JPY_SESSION_NAME"):
        return os.environ["JPY_SESSION_NAME"]
    return sys.argv[0] if sys.argv and sys.argv[0] else None


def _auto_tags(member: str, topic: str) -> dict[str, str]:
    tags = {"member": member, "topic": topic, "source": "notebook", "host": socket.gethostname()}
    sha = _git("rev-parse", "HEAD")
    if sha:
        status = _git("status", "--porcelain", "--untracked-files=no")
        tags.update({"git_sha": sha, "mlflow.source.git.commit": sha})
        if status is not None:
            tags["git_dirty"] = str(bool(status)).lower()
    src = _source()
    if src:
        root = _git("rev-parse", "--show-toplevel")
        try:
            if root and Path(src).is_absolute():
                src = str(Path(src).resolve().relative_to(Path(root).resolve()))
        except ValueError:
            pass
        tags.update({"source_path": src, "mlflow.source.name": src})
        tags["mlflow.source.type"] = "NOTEBOOK" if src.endswith(".ipynb") else "LOCAL"
    try:
        import getpass

        tags["mlflow.user"] = getpass.getuser()
    except Exception:
        pass
    return tags


class NotebookRun:
    """Handle yielded by :func:`notebook_run`. Every method is a no-op when tracking is off, and
    MLflow errors become one warning (then the run turns into a no-op); they never raise."""

    def __init__(self, client: Any = None, run_id: str | None = None, uri: str | None = None):
        self._client, self.run_id, self.uri = client, run_id, uri
        self.experiment_id: str | None = None

    @property
    def active(self) -> bool:
        return self._client is not None and self.run_id is not None

    @property
    def url(self) -> str | None:
        if not (self.active and self.uri and self.uri.startswith("http")):
            return None
        return f"{self.uri.rstrip('/')}/#/experiments/{self.experiment_id}/runs/{self.run_id}"

    def _call(self, what: str, fn, *args, **kwargs) -> None:
        if not self.active:
            return
        try:
            fn(*args, **kwargs)
        except Exception as e:  # never break a notebook because of MLflow
            _warn_once(
                f"run:{self.run_id}",
                f"MLflow {what} failed ({type(e).__name__}: {e}); tracking off for this run.",
            )
            self._client = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        from mlflow.entities import Param

        items = [Param(_key(k), _str(v, _MAX_PARAM)) for k, v in _flatten(params).items()]
        for i in range(0, len(items), _BATCH):
            self._call(
                "log_params", self._client.log_batch, self.run_id, params=items[i : i + _BATCH]
            )

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        from mlflow.entities import RunTag

        items = [RunTag(_key(k), _str(v, _MAX_TAG)) for k, v in _flatten(tags).items()]
        for i in range(0, len(items), _BATCH):
            self._call("set_tags", self._client.log_batch, self.run_id, tags=items[i : i + _BATCH])

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Nested dicts become ``a.b`` keys; non-numeric, NaN and inf values are skipped."""
        if not self.active:
            return
        import time

        from mlflow.entities import Metric

        ts = int(time.time() * 1000)
        items = [
            Metric(_key(k), f, ts, step or 0)
            for k, v in _flatten(metrics).items()
            if (f := _number(v)) is not None
        ]
        for i in range(0, len(items), 1000):
            self._call(
                "log_metrics", self._client.log_batch, self.run_id, metrics=items[i : i + 1000]
            )

    # Artifacts are not logged (quality metrics only): these stay so that an old cell still runs,
    # but they only print a one-time notice.
    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        _no_artifacts()

    def log_dict(self, d: Any, artifact_file: str) -> None:
        _no_artifacts()

    def log_text(self, text: str, artifact_file: str) -> None:
        _no_artifacts()

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        _no_artifacts()

    def log_table(self, data: Any, artifact_file: str) -> None:
        _no_artifacts()


@contextmanager
def notebook_run(
    member: str,
    topic: str,
    run_name: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    uri: str | None = None,
) -> Generator[NotebookRun, None, None]:
    """One MLflow run in experiment ``<member>/<topic>``.

    >>> with notebook_run(member="me", topic="bank", run_name="v2", params={"bank": "v2"}) as run:
    ...     run.log_metrics({"test/far": 0.03, "test/frr": 0.1})

    URI: ``uri`` or ``MLFLOW_TRACKING_URI`` or ``http://localhost:5000``. With ``IML_TRACKING=0``
    or an unreachable server the yielded run is a no-op. An exception in the block marks the run
    FAILED and propagates; MLflow's own errors never do.
    """
    uri = uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_URI
    run = NotebookRun(uri=uri)
    if tracking_disabled():
        yield run
        return
    if not server_reachable(uri):
        _warn_once(
            uri,
            f"MLflow server {uri} is unreachable: run '{run_name}' is NOT tracked "
            "(start it with `make mlflow-up` in the repo root; IML_TRACKING=0 silences this).",
        )
        yield run
        return
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=uri)
        name = f"{member}/{topic}"
        exp = client.get_experiment_by_name(name)
        if exp is None:
            try:
                exp_id = client.create_experiment(name)
            except Exception:  # created concurrently by another process
                exp_id = client.get_experiment_by_name(name).experiment_id
        else:
            exp_id = exp.experiment_id
        if exp is not None and exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp_id)
        if exp is None or EXPERIMENT_KIND_TAG not in exp.tags:
            client.set_experiment_tag(exp_id, EXPERIMENT_KIND_TAG, EXPERIMENT_KIND)
        r = client.create_run(
            exp_id,
            run_name=run_name,
            tags={k: _str(v, _MAX_TAG) for k, v in _auto_tags(member, topic).items()},
        )
        run = NotebookRun(client, r.info.run_id, uri)
        run.experiment_id = exp_id
    except Exception as e:
        _warn_once(f"start:{uri}", f"MLflow run could not be started ({type(e).__name__}: {e}).")
        yield run
        return
    if tags:
        run.set_tags(tags)
    if params:
        run.log_params(params)
    status = "FINISHED"
    try:
        yield run
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    except BaseException:
        status = "FAILED"
        raise
    finally:
        if run.active:
            run._call("set_terminated", run._client.set_terminated, run.run_id, status)
