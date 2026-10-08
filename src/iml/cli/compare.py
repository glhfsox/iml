"""``iml-compare [runs/glhfsox/bn_before_vs_after] [--out comparison]``

Text + table + plot comparison of all runs under a folder (PDF use case 5):
  <out>/table.md / table.csv   one row per run: config diff + metrics per split
  <out>/loss.png, val_metrics.png   overlaid curves, one line per run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iml.runs import RUNS_ROOT, RunRecord, find_runs  # noqa: E402

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


def main(argv=None) -> None:
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
    md = df.to_markdown(index=False, floatfmt=".3f")
    (out / "table.md").write_text(md)
    plot_curves(runs, out)
    print(md)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
