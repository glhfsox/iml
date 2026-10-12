"""``iml-data build --sources sources.csv --out DIR`` / ``iml-data validate DIR``

``build`` turns raw recordings into a dataset that follows DATA_CONTRACT.md (needs the ``data``
dependency group for non-WAV sources); ``validate`` checks any dataset folder against the contract.
"""

from __future__ import annotations

import argparse
import sys

from iml.data.contract import fingerprint, validate_dataset


def main(argv=None) -> None:
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
        from iml.data.preprocess import build

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


if __name__ == "__main__":
    main()
