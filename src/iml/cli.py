"""Command-line entry points, installed by uv as ``iml-train``, ``iml-sweep``, ``iml-compare``,
``iml-export`` and ``iml-data`` (``<command>_main(argv=None)`` each), plus the data helpers they
share with the notebooks (``load_splits``, ``check_run_data``, ...).

``iml-train --config app/models/configs/resnet18.json --data DIR [--set lr=0.01 --set epochs=5]``
    Runs one experiment headlessly (tmux/nohup on the server). The run folder is printed
    at the start so it can be found even if the terminal is lost.

``iml-sweep --config app/models/configs/convnext_tiny.json``
``--grid app/models/configs/grids/lr.json --data DIR``
    Grid file = ``{"field": [values...], ...}``; every combination becomes one run under the
    same member/topic. Finished runs (metrics.json present) are skipped, so a sweep can be
    restarted after an interruption.

``iml-compare [runs/<member>/<arch>] [--out comparison]``
    Text + table + plot comparison of all runs under a folder (PDF use case 5):
      <out>/table.csv   one row per run: config diff + metrics per split
      <out>/loss.png, val_metrics.png   overlaid curves, one line per run

``iml-export --run runs/<member>/<topic>/<run> [--out model.onnx] [--opset 17] [--no-check]`` or
``iml-export --pretrained <model> --out app/models/speaker_verification/<model>/model.onnx``
    Re-exports a run's ``best.pt`` to ONNX by hand (``fit`` already does this at the end of every
    run, see ``iml.export``) and verifies with onnxruntime that the ONNX output matches PyTorch.

``iml-data build --sources sources.csv --out DIR`` / ``iml-data validate DIR``
    ``build`` turns raw recordings into a dataset that follows DATA_CONTRACT.md (needs the ``data``
    dependency group for non-WAV sources); ``validate`` checks any dataset folder against the
    contract.

How the commands obtain data: a dataset folder that follows DATA_CONTRACT.md.
The dataset folder (``--data DIR`` or ``$IML_DATA``) holds ``train/val/test.csv`` with the columns
``path,allowed`` and the 3 s clips as ``.npy`` int16 arrays (contract v3). The manifests are the
split membership, so a run records the dataset fingerprint in ``data.json`` next to its results,
and a finished run is never silently reused for other data.

Since v2 the contract allows single-class datasets (e.g. an enrollment bank of allowed speakers
only), but training does not: FAR/FRR/EER and the decision threshold need both classes, so
``load_splits`` refuses a dataset whose train or val split lacks one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
import pandas as pd
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iml.config import ExperimentConfig  # noqa: E402
from iml.data import (  # noqa: E402
    CLIP_SAMPLES,
    SPLITS,
    SpectrogramDataset,
    fingerprint,
    make_loaders,
    read_split,
    validate_dataset,
)
from iml.export import check_onnx, export_onnx, export_pretrained, export_verified
from iml.runs import RUNS_ROOT, RunRecord, find_runs, load_json, run_dir, save_json  # noqa: E402
from iml.train import fit, get_device  # noqa: E402

__all__ = [
    # entry points
    "train_main",
    "sweep_main",
    "compare_main",
    "export_main",
    "data_main",
    # data helpers (also used by notebooks)
    "resolve_data_dir",
    "TRAINING_SPLITS",
    "check_training_classes",
    "load_splits",
    "data_fingerprint",
    "check_run_data",
    "record_run_data",
    # iml-train / iml-sweep / iml-compare helpers
    "train_parse_args",
    "expand_grid",
    "METRICS",
    "summary_table",
    "plot_curves",
    # re-exported from iml.export (as iml-export's module did)
    "export_onnx",
    "check_onnx",
]


# =================================================================================================
# data: how the commands obtain data (see the module docstring)
# =================================================================================================


def resolve_data_dir(data_dir: str | Path | None) -> Path:
    d = data_dir or os.environ.get("IML_DATA")
    if not d:
        raise ValueError("no dataset: pass --data DIR or set $IML_DATA")
    if not Path(d).is_dir():
        raise FileNotFoundError(f"dataset folder not found: {d}")
    return Path(d)


# train: the loss; val: checkpoint selection and the decision threshold (FAR == FRR point)
TRAINING_SPLITS = ("train", "val")


def check_training_classes(data_dir: Path, items: dict[str, list[tuple[Path, int]]]) -> None:
    """Fail if a split used for training or threshold selection lacks a class; warn for test."""
    for split in SPLITS:
        labels = {a for _, a in items[split]}
        if labels == {0, 1}:
            continue
        has = f"only allowed={labels.pop()}" if labels else "no clips"
        if split in TRAINING_SPLITS:
            raise ValueError(
                f"dataset {data_dir}: split {split} has {has}; training needs both classes "
                "(the contract allows single-class datasets since v2, e.g. for enrollment banks)"
            )
        warnings.warn(
            f"dataset {data_dir}: split {split} has {has}; its FAR/FRR/EER will be NaN",
            stacklevel=3,
        )


def load_splits(cfg: ExperimentConfig, data_dir: str | Path | None = None) -> dict[str, DataLoader]:
    """Loaders for train/val/test of a validated contract dataset that has both classes in train
    and val (a single-class test split only warns)."""
    d = resolve_data_dir(data_dir)
    errors = validate_dataset(d, check_audio=False)
    if errors:
        raise ValueError(f"{d} violates the data contract:\n  " + "\n  ".join(errors))
    items = {s: read_split(d, s) for s in SPLITS}
    check_training_classes(d, items)
    if int(cfg.excerpt_sec * cfg.sample_rate) != CLIP_SAMPLES:
        warnings.warn(
            f"excerpt_sec={cfg.excerpt_sec} != the 3 s contract clips: the model sees "
            "crops/padding of each clip (re-run `make configs` if this is unintended)",
            stacklevel=2,
        )
    datasets = {s: SpectrogramDataset(items[s], cfg, train=s == "train") for s in SPLITS}
    return make_loaders(datasets, cfg)


def data_fingerprint(data_dir: str | Path | None = None) -> dict:
    return fingerprint(resolve_data_dir(data_dir))


def check_run_data(out: Path, fp: dict, overwrite: bool) -> None:
    """Refuse to reuse a finished run that was trained on different data."""
    done, rec = out / "metrics.json", out / "data.json"
    if overwrite or not done.exists():
        return
    if not rec.exists():
        raise RuntimeError(
            f"{out} is finished but has no data.json (trained before data tracking, data unknown); "
            "pass --overwrite to retrain it"
        )
    old = load_json(rec)
    if old.get("sha1") != fp.get("sha1"):
        raise RuntimeError(
            f"{out} was trained on different data ({old}); pass --overwrite to retrain it"
        )


def record_run_data(out: Path, fp: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_json(fp, out / "data.json")


# =================================================================================================
# iml-train: one configuration
# =================================================================================================


def train_parse_args(argv=None):
    p = argparse.ArgumentParser(description="train one configuration")
    p.add_argument("--config", required=True, help="path to an ExperimentConfig json")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field (repeatable)",
    )
    p.add_argument("--member", help="override cfg.member")
    p.add_argument("--topic", help="override cfg.topic")
    p.add_argument("--runs-root", default=str(RUNS_ROOT))
    p.add_argument("--device", help="hip (AMD, = cuda in torch) | cuda | mps | cpu (default: auto)")
    p.add_argument("--data", help="dataset folder (DATA_CONTRACT.md); default: $IML_DATA")
    p.add_argument(
        "--overwrite", action="store_true", help="retrain even if the run folder is finished"
    )
    return p.parse_args(argv)


def train_main(argv=None) -> None:
    args = train_parse_args(argv)
    cfg = ExperimentConfig.from_json(args.config).with_overrides(*args.set)
    if args.member:
        cfg = cfg.with_overrides(f"member={args.member}")
    if args.topic:
        cfg = cfg.with_overrides(f"topic={args.topic}")
    out = run_dir(cfg, args.runs_root)
    fp = data_fingerprint(args.data)
    check_run_data(out, fp, args.overwrite)
    trains = args.overwrite or not (out / "metrics.json").exists()
    loaders = load_splits(cfg, data_dir=args.data)
    fit(cfg, loaders, root=args.runs_root, device=get_device(args.device), overwrite=args.overwrite)
    if trains:
        record_run_data(out, fp)


# =================================================================================================
# iml-sweep: a grid of configurations
# =================================================================================================


def expand_grid(base: ExperimentConfig, grid: dict[str, list]) -> list[ExperimentConfig]:
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        d = base.to_dict()
        d.update(dict(zip(keys, values, strict=True)))
        configs.append(ExperimentConfig.from_dict(d))
    return configs


def sweep_main(argv=None) -> None:
    p = argparse.ArgumentParser(description="run a grid of configurations")
    p.add_argument("--config", required=True)
    p.add_argument("--grid", required=True, help="json: {field: [values]}")
    p.add_argument("--member")
    p.add_argument("--topic")
    p.add_argument("--runs-root", default=str(RUNS_ROOT))
    p.add_argument("--device", help="hip | cuda | mps | cpu (default: auto)")
    p.add_argument("--data", help="dataset folder (DATA_CONTRACT.md); default: $IML_DATA")
    p.add_argument("--overwrite", action="store_true", help="retrain finished runs")
    args = p.parse_args(argv)

    base = ExperimentConfig.from_json(args.config)
    for key, val in (("member", args.member), ("topic", args.topic)):
        if val:
            base = base.with_overrides(f"{key}={val}")
    configs = expand_grid(base, json.loads(Path(args.grid).read_text()))
    device = get_device(args.device)
    fp = data_fingerprint(args.data)
    print(f"{len(configs)} configurations")
    for i, cfg in enumerate(configs, 1):
        out = run_dir(cfg, args.runs_root)
        check_run_data(out, fp, overwrite=args.overwrite)
        if (out / "metrics.json").exists() and not args.overwrite:
            print(f"[{i}/{len(configs)}] skip (done): {out.name}")
            continue
        print(f"[{i}/{len(configs)}] {out.name}")
        loaders = load_splits(cfg, data_dir=args.data)
        fit(cfg, loaders, root=args.runs_root, device=device, overwrite=args.overwrite)
        record_run_data(out, fp)


# =================================================================================================
# iml-compare: table + curves of finished runs
# =================================================================================================

METRICS = ("macro_f1", "far", "frr", "eer", "accuracy")


def summary_table(runs: list[RunRecord], splits=("val", "test")) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {
            "run": r.name,
            "model": r.cfg.model,
            "params": r.metrics.get("params"),
            "best_epoch": r.metrics.get("best_epoch"),
        }
        row.update(
            {k: v for k, v in r.cfg.diff_from_default().items() if not isinstance(v, dict | list)}
        )
        for s in splits:
            if s in r.metrics:
                row.update({f"{s}_{m}": r.metrics[s].get(m) for m in METRICS})
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_key = "val_macro_f1" if "val_macro_f1" in df else "run"
    return df.sort_values(sort_key, ascending=False).reset_index(drop=True)


def plot_curves(runs: list[RunRecord], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        ax.plot(r.history["epoch"], r.history["val_loss"], label=r.name)
    ax.set_xlabel("epoch"), ax.set_ylabel("val loss"), ax.grid(alpha=0.3), ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "loss.png", dpi=120), plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, m in zip(axes, ("val_macro_f1", "val_far", "val_frr"), strict=True):
        for r in runs:
            ax.plot(r.history["epoch"], r.history[m], label=r.name)
        ax.set_title(m), ax.set_ylim(0, 1), ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "val_metrics.png", dpi=120), plt.close(fig)


def compare_main(argv=None) -> None:
    p = argparse.ArgumentParser(description="compare finished runs")
    p.add_argument("root", nargs="?", default=str(RUNS_ROOT), help="folder to scan recursively")
    p.add_argument("--out", help="where to write table/plots (default: <root>/comparison)")
    args = p.parse_args(argv)
    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"no finished runs under {args.root}")
    out = Path(args.out or Path(args.root) / "comparison")
    out.mkdir(parents=True, exist_ok=True)
    df = summary_table(runs)
    df.to_csv(out / "table.csv", index=False)
    plot_curves(runs, out)
    print(df.to_markdown(index=False, floatfmt=".3f"))
    print(f"\nwritten to {out}")


# =================================================================================================
# iml-export: best.pt -> verified model.onnx by hand
# =================================================================================================


def export_main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="export a run's best model, or a registry model's pretrained starting point, "
        "to ONNX and verify it"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run folder or .pt file")
    src.add_argument(
        "--pretrained",
        metavar="MODEL",
        help="registry model: ImageNet backbone + fresh head, threshold 0.5 (make init-models)",
    )
    p.add_argument(
        "--out", help="output path (default: <run>/model.onnx; required with --pretrained)"
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args(argv)
    if args.pretrained:
        if not args.out:
            p.error("--pretrained needs --out")
        out = Path(args.out)
        _, model, _ = export_pretrained(args.pretrained, out, args.opset)  # also for --no-check
        ref: dict = {"model": model}

        def write(tmp: Path) -> Path:
            return export_pretrained(args.pretrained, tmp, args.opset)[0]

    else:
        run = Path(args.run)
        out = (
            Path(args.out)
            if args.out
            else (run if run.suffix == ".pt" else run / "best.pt").parent / "model.onnx"
        )
        ref = {"run": args.run}

        def write(tmp: Path) -> Path:
            return export_onnx(args.run, tmp, args.opset)

    if args.no_check:
        out = write(out)
        print(f"written {out} ({out.stat().st_size / 1e6:.2f} MB)")
        return
    diff = export_verified(out, write, **ref)
    print(f"written {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"onnxruntime matches pytorch (max abs diff {diff:.2e})")


# =================================================================================================
# iml-data: build / validate a contract dataset
# =================================================================================================


def data_main(argv=None) -> None:
    p = argparse.ArgumentParser(description="build / validate a contract dataset")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="raw recordings -> train/val/test.csv + 3 s clips")
    b.add_argument(
        "--sources", required=True, help="sources.csv (path,speaker,allowed) next to recordings/"
    )
    b.add_argument("--out", required=True, help="output dataset folder")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--overwrite", action="store_true", help="replace an existing dataset in --out")
    v = sub.add_parser("validate", help="check a dataset folder against DATA_CONTRACT.md")
    v.add_argument("data_dir")
    args = p.parse_args(argv)

    if args.cmd == "build":
        from iml.preprocess import build

        report = build(args.sources, args.out, seed=args.seed, overwrite=args.overwrite)
        n_dropped = len(report["dropped"])
        print(f"built {args.out}: clips {report['clips']}, dropped recordings {n_dropped}")
        target = args.out
    else:
        target = args.data_dir
    errors = validate_dataset(target)
    if errors:
        print(f"{len(errors)} contract violation(s):", *errors, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print("contract OK:", fingerprint(target)["counts"])
