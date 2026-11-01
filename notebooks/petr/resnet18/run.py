"""Reproduce ResNet-18 experiments and export compact evidence for Git."""

import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "runs/petr/resnet18_lr"
DATA = "data/training_dataset_v2"


def run(*args):
    print("RUN", *args, flush=True)
    subprocess.run(list(args), cwd=ROOT, check=True)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def records():
    found = []
    for path in sorted(RUNS.glob("*/metrics.json")):
        config = read(path.parent / "config.json")
        metrics = read(path)
        found.append(
            {"path": str(path.parent.relative_to(ROOT)), "config": config, "metrics": metrics}
        )
    return found


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Dataset folder matching dataset_snapshot.json")
    args = parser.parse_args()
    receipt_path = HERE / "dataset_snapshot.json"
    receipt = read(receipt_path) if receipt_path.exists() else None
    if args.data:
        DATA = args.data
    elif receipt and Path(receipt["local_snapshot"]).exists():
        DATA = receipt["local_snapshot"]
    from iml.cli import data_fingerprint

    fingerprint = data_fingerprint(ROOT / DATA)
    if receipt and fingerprint["sha1"] != receipt["fingerprint"]["sha1"]:
        raise ValueError("Dataset changed: use the recorded snapshot or a new experiment topic")
    run("uv", "run", "iml-data", "validate", DATA)
    base = str((HERE / "baseline.json").relative_to(ROOT))
    run(
        "uv",
        "run",
        "iml-train",
        "--config",
        base,
        "--data",
        DATA,
        "--topic",
        "resnet18_smoke",
        "--set",
        "epochs=1",
    )
    run(
        "uv",
        "run",
        "iml-sweep",
        "--config",
        base,
        "--grid",
        str((HERE / "lr_grid.json").relative_to(ROOT)),
        "--data",
        DATA,
    )
    trials = [r for r in records() if r["config"]["seed"] == 42]
    if len(trials) != 3:
        raise RuntimeError("Expected three completed learning-rate trials")
    selected = min(
        trials,
        key=lambda r: (r["metrics"]["val"]["eer"], r["metrics"]["val"]["loss"], r["config"]["lr"]),
    )
    write(HERE / "repeat_config.json", selected["config"])
    run(
        "uv",
        "run",
        "iml-sweep",
        "--config",
        str((HERE / "repeat_config.json").relative_to(ROOT)),
        "--grid",
        str((HERE / "seed_grid.json").relative_to(ROOT)),
        "--data",
        DATA,
    )
    run("uv", "run", "iml-compare", str(RUNS.relative_to(ROOT)))
    run("uv", "run", "iml-export", "--run", selected["path"])
    all_runs = records()
    if len(all_runs) != 5:
        raise RuntimeError("Expected five completed main runs")
    for record in all_runs:
        recorded = read(ROOT / record["path"] / "data.json")
        if recorded["sha1"] != fingerprint["sha1"]:
            raise ValueError("Run dataset fingerprint mismatch")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    for record in all_runs:
        config = record["config"]
        dst = out / ("lr_" + str(config["lr"]) + "_seed_" + str(config["seed"]))
        dst.mkdir(exist_ok=True)
        for name in [
            "config.json",
            "metrics.json",
            "history.csv",
            "data.json",
            "env.json",
            "loss.png",
            "metrics.png",
        ]:
            src = ROOT / record["path"] / name
            if src.exists():
                shutil.copy2(src, dst / name)
    shutil.copytree(RUNS / "comparison", out / "comparison", dirs_exist_ok=True)
    repeats = [r for r in all_runs if r["config"]["lr"] == selected["config"]["lr"]]
    report = {
        "selection_rule": "Choose by validation EER, then validation loss; seed=42 only.",
        "dataset": DATA,
        "selected_run": selected["path"],
        "selected_config": selected["config"],
        "selected_metrics": selected["metrics"],
        "repeat_validation_eer_mean": statistics.mean(r["metrics"]["val"]["eer"] for r in repeats),
        "repeat_validation_eer_std": statistics.stdev(r["metrics"]["val"]["eer"] for r in repeats),
        "runs": [
            {
                "lr": r["config"]["lr"],
                "seed": r["config"]["seed"],
                "epochs_run": r["metrics"]["epochs_run"],
                "best_epoch": r["metrics"]["best_epoch"],
                "val_eer": r["metrics"]["val"]["eer"],
                "val_loss": r["metrics"]["val"]["loss"],
                "path": r["path"],
            }
            for r in all_runs
        ],
    }
    write(out / "summary.json", report)
    (HERE / "selected_run.txt").write_text(selected["path"] + "\n")
    print("FINAL_SUMMARY", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
