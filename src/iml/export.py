"""ONNX export: ``export_onnx`` writes a run's ``model.onnx`` next to ``best.pt``,
``export_pretrained`` writes the stable starting point of a registry model (ImageNet backbone +
fresh 2-class head, threshold 0.5) for ``app/models/speaker_verification/<model>/``, and
``check_onnx`` verifies a graph against PyTorch with onnxruntime (two time lengths, batch 3).

``fit`` calls both at the end of every run (``cfg.export_onnx``), so a finished run folder always
holds a verified graph; ``iml-export`` re-runs them by hand. The graph has dynamic batch and time
axes and is rebuilt from the run's ``config.json``, so it works for every model in the registry.

The graph is self-describing: its ``metadata_props`` hold the run's config (``iml.config``, JSON),
the val-chosen decision threshold (``iml.decision_threshold``), the model name (``iml.model``) and
the data contract version (``iml.contract_version``); ``read_onnx_metadata`` reads them back, so a
frozen model folder needs nothing but ``model.onnx``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import torch

from iml.config import ExperimentConfig
from iml.data import CONTRACT_VERSION, n_frames
from iml.models import build_model
from iml.runs import load_checkpoint

__all__ = [
    "export_onnx",
    "export_pretrained",
    "check_onnx",
    "read_onnx_metadata",
    "load_run_model",
    "onnx_session",
    "write_onnx",
    "export_verified",
]

_CONFIG_KEY = "iml.config"
_THRESHOLD_KEY = "iml.decision_threshold"


def load_run_model(run: str | Path) -> tuple[torch.nn.Module, ExperimentConfig, float]:
    """``best.pt`` of a run folder (or a ``.pt`` file) -> eval model, its config, its threshold."""
    path = Path(run)
    ckpt = load_checkpoint(path if path.suffix == ".pt" else path / "best.pt")
    cfg = ckpt["config"]
    model = build_model(cfg, load_pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    threshold = float(ckpt.get("metrics", {}).get("decision_threshold", 0.5))
    return model.eval(), cfg, threshold


def write_onnx(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    out: str | Path,
    threshold: float = 0.5,
    opset: int = 17,
) -> Path:
    """Export ``model`` (eval, on CPU) to ``out`` with the iml metadata (config, threshold)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    example = torch.randn(1, 1, cfg.n_mels, n_frames(cfg))
    torch.onnx.export(
        model,
        example,
        str(out),
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    graph = onnx.load(str(out))
    meta = {
        _CONFIG_KEY: json.dumps(cfg.to_dict()),
        _THRESHOLD_KEY: str(float(threshold)),
        "iml.model": cfg.model,
        "iml.contract_version": str(CONTRACT_VERSION),
    }
    kept = [p for p in graph.metadata_props if p.key not in meta]
    del graph.metadata_props[:]
    graph.metadata_props.extend(kept)
    for key, value in meta.items():
        graph.metadata_props.add(key=key, value=value)
    onnx.save(graph, str(out))
    return out


def export_onnx(run: str | Path, out: str | Path | None = None, opset: int = 17) -> Path:
    """A run's ``best.pt`` -> ``model.onnx`` (default: next to it) with its config and threshold."""
    path = Path(run)
    ckpt_path = path if path.suffix == ".pt" else path / "best.pt"
    model, cfg, threshold = load_run_model(ckpt_path)
    return write_onnx(
        model, cfg, Path(out) if out else ckpt_path.parent / "model.onnx", threshold, opset
    )


def export_verified(
    out: str | Path,
    write: Callable[[Path], Path],
    model: torch.nn.Module | None = None,
    run: str | Path | None = None,
) -> float:
    """Write the graph to ``<out>.tmp`` with ``write``, verify it against ``model``/``run`` and
    only then give it its final name; on any failure no graph is left behind. Returns the diff."""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    try:
        write(tmp)
        diff = check_onnx(tmp, run=run, model=model)
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return diff


def export_pretrained(
    name: str, out: str | Path, opset: int = 17
) -> tuple[Path, torch.nn.Module, ExperimentConfig]:
    """The stable starting point of registry model ``name``: ImageNet-pretrained backbone with a
    fresh 2-class head (seeded), threshold 0.5, the baseline config in the metadata. This is what
    ``make init-models`` puts into ``app/models/speaker_verification/<name>/model.onnx``."""
    if name == "custom":
        raise ValueError("custom has no pretrained weights: train it and use `make freeze`")
    cfg = ExperimentConfig(model=name)  # pretrained=True by default
    torch.manual_seed(cfg.seed)
    model = build_model(cfg).eval()
    return write_onnx(model, cfg, out, 0.5, opset), model, cfg


def onnx_session(path: str | Path, providers: tuple[str, ...] = ("CPUExecutionProvider",)):
    """onnxruntime session with an explicit thread count (otherwise ORT pins threads to cores and
    logs an affinity error per thread inside containers with a restricted CPU set)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    # sched_getaffinity is Linux-only; macOS falls back to cpu_count
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    so.intra_op_num_threads = max(1, cpus or 1)
    so.log_severity_level = 3  # errors only
    return ort.InferenceSession(str(path), sess_options=so, providers=list(providers))


def read_onnx_metadata(path: str | Path) -> tuple[ExperimentConfig, float]:
    """Config and decision threshold stored in a graph written by :func:`export_onnx`."""
    graph = onnx.load(str(path), load_external_data=False)
    meta = {p.key: p.value for p in graph.metadata_props}
    if _CONFIG_KEY not in meta:
        raise ValueError(f"{path} has no iml metadata; re-export it with `iml-export --run <run>`")
    cfg = ExperimentConfig.from_dict(json.loads(meta[_CONFIG_KEY]))
    return cfg, float(meta.get(_THRESHOLD_KEY, 0.5))


def check_onnx(
    onnx_path: str | Path,
    run: str | Path | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-4,
    model: torch.nn.Module | None = None,
) -> float:
    """Max abs difference between PyTorch and onnxruntime logits on random input (2 lengths).

    The reference is ``model`` (eval, CPU) or the ``best.pt`` of ``run``; the config comes from the
    graph's metadata. Fails when a length's diff exceeds ``atol + rtol * max|logits|``."""
    cfg, _ = read_onnx_metadata(onnx_path)
    if model is None:
        if run is None:
            raise ValueError("check_onnx needs a run folder or a model to compare against")
        model, _, _ = load_run_model(run)
    model = model.eval().cpu()
    sess = onnx_session(onnx_path)
    worst, failed = 0.0, None
    for frames in (n_frames(cfg), n_frames(cfg) // 2 + 1):
        x = torch.randn(3, 1, cfg.n_mels, frames)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(["logits"], {"spectrogram": x.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        tol = atol + rtol * float(np.abs(ref).max())
        worst = max(worst, diff)
        if diff > tol and failed is None:
            failed = (diff, tol)
    if failed is not None:
        raise RuntimeError(
            f"ONNX output differs from PyTorch by {failed[0]:.2e} (> {failed[1]:.2e})"
        )
    return worst
