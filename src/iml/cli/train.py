"""``iml-train --config configs/cnn10.json [--set lr=0.01 --set epochs=5] [--dummy]``

Runs one experiment headlessly (tmux/nohup on the server). The run folder is printed
at the start so it can be found even if the terminal is lost.
"""

from __future__ import annotations

import argparse

from iml.cli._data import load_splits
from iml.config import ExperimentConfig
from iml.device import get_device
from iml.runs import RUNS_ROOT
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
    p.add_argument("--dummy", action="store_true", help="synthetic data for smoke tests")
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
    loaders = load_splits(cfg, dummy=args.dummy)
    fit(cfg, loaders, root=args.runs_root, device=get_device(args.device), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
