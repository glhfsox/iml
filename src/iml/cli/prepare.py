"""Build a training manifest from manually reviewed Telegram collector metadata."""

from __future__ import annotations

import argparse
import json

from iml.data.manifest import prepare_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="prepare reviewed Telegram audio for training")
    parser.add_argument("--metadata", required=True, help="collector metadata.csv after review")
    parser.add_argument(
        "--source-root",
        required=True,
        help="collector project root used to resolve normalized_audio_path",
    )
    parser.add_argument("--target-speaker", required=True, help="speaker_ref mapped to class 1")
    parser.add_argument("--output", default="data/processed", help="gitignored output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.6)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report = prepare_dataset(
        metadata_csv=args.metadata,
        source_root=args.source_root,
        output_dir=args.output,
        target_speaker=args.target_speaker,
        seed=args.seed,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"training manifest: {args.output}/splits.json")


if __name__ == "__main__":
    main()
