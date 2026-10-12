"""``iml-train --config configs/cnn10.json --data DIR [--set lr=0.01 --set epochs=5]``

Runs one experiment headlessly (tmux/nohup on the server). The run folder is printed
at the start so it can be found even if the terminal is lost.
"""

from __future__ import annotations

import argparse

from iml.cli._data import check_run_data, data_fingerprint, load_splits, record_run_data
from iml.config import ExperimentConfig
from iml.device import get_device
from iml.runs import RUNS_ROOT, run_dir
from iml.train import fit


def parse_args(argv=None):
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


def main(argv=None) -> None:
    args = parse_args(argv)
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


if __name__ == "__main__":
    main()
