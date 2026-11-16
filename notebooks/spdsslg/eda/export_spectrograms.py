"""Draw model-input spectrograms and save them as PNG pictures.

Run from the repository root in the project environment::

    python -m iml.export_spectrograms --data data/training_dataset_v2
    python -m iml.export_spectrograms --data data/training_dataset_v3 --out /tmp/v3-pictures
    python -m iml.export_spectrograms --data data/training_dataset_v2 --config path/to/config.json

The default output is DATA/spectrograms. Each PNG is grouped by split/class and
preserves the input clip's relative path. spectrogram_index.csv maps images to
clips; export_metadata.json records the frontend settings. Each run redraws the
selected images, including existing ones, so changed inputs/settings take effect.

Manifest parsing and validation come from iml.contract. SpectrogramDataset computes
exactly the evaluation input tensor: log-mel power, dB conversion, then configured
normalization. Native three-second clips need no cropping. Only the rendered PNG
uses a colormap; the model receives the numerical tensor, not this image.
Pictures share fixed color limits within each normalization mode, including
across datasets. Values outside the display range saturate without changing
the tensor. Raw dB, scaled dB, and per-sample units remain different quantities.

The Python API returns the clip-to-image index as a DataFrame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from iml.config import ExperimentConfig
from iml.contract import SPLITS, read_split, validate_dataset
from iml.data import SpectrogramDataset

FRONTEND_FIELDS = ("sample_rate", "excerpt_sec", "n_fft", "hop_length", "n_mels", "normalize_input")

# Display ranges only. "global" applies (dB + 40) / 40 to the raw dB limits.
COLOR_LIMITS = {"none": (-80.0, 40.0), "global": (-1.0, 2.0), "per_sample": (-3.0, 5.0)}


def draw_spectrogram(ax, spectrogram, cfg: ExperimentConfig, *, title: str = ""):
    """Draw one (mel bands, time frames) tensor with correctly labelled axes.

    Returns the matplotlib image so callers can add a shared or separate colorbar.
    Mel bands are not equally spaced in Hz, so the vertical axis uses band indices.
    """
    values = np.asarray(spectrogram)
    if values.ndim != 2 or values.shape[0] != cfg.n_mels:
        raise ValueError("Expected a two-dimensional (mel bands, frames) spectrogram")
    hop_seconds = cfg.hop_length / cfg.sample_rate
    low, high = COLOR_LIMITS[cfg.normalize_input]
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=(
            -hop_seconds / 2,
            (values.shape[1] - 0.5) * hop_seconds,
            -0.5,
            cfg.n_mels - 0.5,
        ),
        vmin=low,
        vmax=high,
    )
    ax.set(xlabel="Time (s)", ylabel="Mel band", title=title, xlim=(0, cfg.excerpt_sec))
    return image


def colorbar_label(cfg: ExperimentConfig) -> str:
    return {
        "per_sample": "Normalized log-mel (standard deviations)",
        "global": "Scaled log-mel",
        "none": "Log-mel power (dB)",
    }[cfg.normalize_input]


def export_spectrograms(
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    cfg: ExperimentConfig | None = None,
    *,
    dpi: int = 120,
    verbose: bool = True,
) -> pd.DataFrame:
    """Draw and save one PNG per clip, replacing existing images at the same paths.

    Input waveforms, labels, and split manifests are never written. Validation
    checks file headers once; each waveform is then loaded by SpectrogramDataset.
    """
    import matplotlib.pyplot as plt

    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else data_dir / "spectrograms"
    cfg = cfg or ExperimentConfig()
    if dpi < 1:
        raise ValueError("DPI must be positive")
    if output_dir in (data_dir, data_dir / "audio"):
        raise ValueError("Choose a dedicated output folder, not the dataset root or audio folder")
    errors = validate_dataset(data_dir)
    if errors:
        raise ValueError(errors)

    rows = []
    with torch.inference_mode():
        for split in SPLITS:
            items = read_split(data_dir, split)
            dataset = SpectrogramDataset(items, cfg, train=False)
            for i, (path, label) in enumerate(
                tqdm(items, desc=f"PNG {split}", disable=not verbose)
            ):
                relative = path.relative_to(data_dir)
                label_name = "allowed" if label == 1 else "invalid"
                image_path = Path(split) / label_name / relative.with_suffix(".png")
                target = output_dir / image_path
                tensor, _ = dataset[i]
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid frontend output for {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
                try:
                    image = draw_spectrogram(
                        ax,
                        tensor.squeeze(0).cpu().numpy(),
                        cfg,
                        title=f"{path.stem}\n{split} · allowed={label}",
                    )
                    fig.colorbar(image, ax=ax, label=colorbar_label(cfg), extend="both")
                    fig.savefig(target, dpi=dpi)
                finally:
                    plt.close(fig)
                rows.append(
                    {
                        "path": relative.as_posix(),
                        "allowed": label,
                        "split": split,
                        "image": image_path.as_posix(),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(rows, columns=["path", "allowed", "split", "image"])
    index.to_csv(output_dir / "spectrogram_index.csv", index=False)
    metadata = {
        "data_dir": str(data_dir),
        "frontend": {k: getattr(cfg, k) for k in FRONTEND_FIELDS},
        "color_limits": COLOR_LIMITS[cfg.normalize_input],
        "dpi": dpi,
        "image_count": len(index),
    }
    (output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if verbose:
        print(f"Saved {len(index)} spectrogram pictures: {output_dir}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="dataset with train/val/test CSVs")
    parser.add_argument("--out", type=Path, help="picture folder (default: DATA/spectrograms)")
    parser.add_argument("--config", type=Path, help="experiment config with frontend settings")
    parser.add_argument("--dpi", type=int, default=120, help="PNG resolution (default: 120)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    export_spectrograms(args.data, args.out, cfg, dpi=args.dpi)


if __name__ == "__main__":
    main()
