"""Export pictures of the same log-mel tensors used by the training loader.

Run from the repository root, using the project environment::

    python -m iml.export_spectrograms --data data/training_dataset_v2
    python -m iml.export_spectrograms --data data/training_dataset_v2 --out /tmp/v2-pictures
    python -m iml.export_spectrograms --data data/training_dataset_v2 --config runs/.../config.json

``--out`` defaults to ``<dataset>/spectrograms``. The input must satisfy
DATA_CONTRACT.md: train/val/test CSVs and 16 kHz, three-second int16 NPY clips.
No model, training, download, or waveform modification is involved.

Defaults match the training frontend: 80 mel bands, FFT 512, hop 160, power=2,
AmplitudeToDB(top_db=80), followed by per-sample normalization. An experiment
``--config`` can supply different frontend settings. The exporter calls
SpectrogramDataset directly, using deterministic evaluation cropping; with the
default three-second clips, training and evaluation crops are identical.

PNG files are placed in ``<out>/<split>/<allowed|invalid>/<original path>.png``.
``spectrogram_index.csv`` maps each picture to its clip and label;
``export_metadata.json`` records frontend settings and the dataset fingerprint.
PNG colors are a visualization of the tensor, not a replacement training input.
Per-sample-normalized pictures share a fixed color range; out-of-range colors
are saturated, and the underlying tensor is unchanged.

An unchanged export can resume by skipping existing PNGs. Different input data,
frontend settings, or DPI require ``--overwrite``. That flag recreates expected
images but never deletes unrelated files. The input fingerprint is checked again
after export. Other datasets are processed only when selected explicitly.
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
from iml.contract import SPLITS, fingerprint, validate_dataset
from iml.data import SpectrogramDataset

FRONTEND_FIELDS = ("sample_rate", "excerpt_sec", "n_fft", "hop_length", "n_mels", "normalize_input")


def draw_spectrogram(ax, spectrogram, cfg: ExperimentConfig, *, title: str = ""):
    """Draw one (mel bands, time frames) tensor with correctly labelled axes.

    Returns the matplotlib image so callers can add a shared or separate colorbar.
    Mel bands are not equally spaced in Hz, so the vertical axis uses band indices.
    """
    values = np.asarray(spectrogram)
    if values.ndim != 2 or values.shape[0] != cfg.n_mels:
        raise ValueError("Expected a two-dimensional (mel bands, frames) spectrogram")
    hop_seconds = cfg.hop_length / cfg.sample_rate
    limits = {"vmin": -3.0, "vmax": 5.0} if cfg.normalize_input == "per_sample" else {}
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
        **limits,
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
    overwrite: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Write one PNG per manifest row and return the clip-to-image index.

    Reuses SpectrogramDataset without changing labels, splits, or input audio.
    ``cfg`` affects only the frontend, not model construction or training.
    """
    import matplotlib.pyplot as plt

    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else data_dir / "spectrograms"
    cfg = cfg or ExperimentConfig()
    if dpi < 1:
        raise ValueError("DPI must be positive")
    if output_dir == data_dir or output_dir == data_dir / "audio":
        raise ValueError("Choose a dedicated output folder, not the dataset root or audio folder")
    errors = validate_dataset(data_dir)
    if errors:
        raise ValueError(errors)
    original = fingerprint(data_dir)
    frontend = {field: getattr(cfg, field) for field in FRONTEND_FIELDS}
    signature = {"dataset_sha1": original["sha1"], "frontend": frontend, "dpi": dpi}
    metadata_path = output_dir / "export_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("signature") != signature and not overwrite:
            raise ValueError(
                "Export settings or dataset changed; choose another folder or --overwrite"
            )
    elif output_dir.exists() and any(output_dir.rglob("*.png")) and not overwrite:
        raise ValueError("Unidentified existing PNGs; choose another folder or --overwrite")

    # Preserve the original path to avoid collisions between identically named clips.
    rows = []
    for split in SPLITS:
        manifest = pd.read_csv(data_dir / f"{split}.csv")
        for row in manifest.itertuples(index=False):
            label_name = "allowed" if int(row.allowed) == 1 else "invalid"
            image = Path(split) / label_name / Path(row.path).with_suffix(".png")
            rows.append(
                {
                    "path": row.path,
                    "allowed": int(row.allowed),
                    "split": split,
                    "image": image.as_posix(),
                }
            )
    index = pd.DataFrame(rows, columns=["path", "allowed", "split", "image"])
    if index.image.duplicated().any():
        raise ValueError("Image paths are not unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "signature": signature,
        "dataset": original,
        "complete": False,
        "normalization_color_limits": [-3, 5] if cfg.normalize_input == "per_sample" else None,
        "crop": "evaluation (center); native three-second clips need no cropping",
        "tensor_source": "iml.data.SpectrogramDataset",
        "torch_version": str(torch.__version__),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    written = skipped = 0
    with torch.inference_mode():
        for split in SPLITS:
            part = index[index.split.eq(split)]
            dataset = SpectrogramDataset(
                [(data_dir / row.path, row.allowed) for row in part.itertuples()], cfg, train=False
            )
            for i, row in enumerate(
                tqdm(part.itertuples(), total=len(part), desc=f"PNG {split}", disable=not verbose)
            ):
                target = output_dir / row.image
                if target.exists() and not overwrite:
                    skipped += 1
                    continue
                tensor, label = dataset[i]
                if label != row.allowed or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid frontend output for {row.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
                try:
                    image = draw_spectrogram(
                        ax,
                        tensor.squeeze(0).cpu().numpy(),
                        cfg,
                        title=f"{Path(row.path).stem}\n{split} · allowed={label}",
                    )
                    fig.colorbar(image, ax=ax, label=colorbar_label(cfg), extend="both")
                    temporary = target.with_suffix(".png.tmp")
                    fig.savefig(temporary, format="png", dpi=dpi)
                    temporary.replace(target)
                    written += 1
                finally:
                    plt.close(fig)
    if fingerprint(data_dir) != original:
        raise RuntimeError("Input data changed during export; the export is marked incomplete")
    index.to_csv(output_dir / "spectrogram_index.csv", index=False)
    metadata.update(complete=True, image_count=len(index), written=written, skipped=skipped)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if verbose:
        print(f"{written} written, {skipped} reused: {output_dir}")
        print("Input manifests and waveform fingerprint unchanged:", original["sha1"])
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset with train/val/test CSVs")
    parser.add_argument("--out", type=Path, help="Output folder (default: DATA/spectrograms)")
    parser.add_argument("--config", type=Path, help="Experiment JSON containing frontend settings")
    parser.add_argument("--dpi", type=int, default=120, help="PNG resolution (default: 120)")
    parser.add_argument("--overwrite", action="store_true", help="Recreate expected images")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    export_spectrograms(args.data, args.out, cfg, dpi=args.dpi, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
