"""``iml-sweep --config configs/simple_cnn.json --grid configs/grids/bn.json --data DIR``

Grid file = ``{"field": [values...], ...}``; every combination becomes one run under the
same member/topic. Finished runs (metrics.json present) are skipped, so a sweep can be
restarted after an interruption.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from iml.cli._data import check_run_data, data_fingerprint, load_splits, record_run_data
from iml.config import ExperimentConfig
from iml.device import get_device
from iml.runs import RUNS_ROOT, run_dir
from iml.train import fit


def expand_grid(base: ExperimentConfig, grid: dict[str, list]) -> list[ExperimentConfig]:
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        d = base.to_dict()
        d.update(dict(zip(keys, values, strict=True)))
        configs.append(ExperimentConfig.from_dict(d))
    return configs


def main(argv=None) -> None:
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


if __name__ == "__main__":
    main()
