"""``iml-export --run runs/<member>/<topic>/<run> [--out model.onnx] [--time-frames 201]``

Converts a run's ``best.pt`` into ONNX (dynamic batch and time axes) and verifies with
onnxruntime that the ONNX output matches PyTorch. Works for every model in the registry,
whatever layers were added or removed, because the architecture is rebuilt from the run's
``config.json``. Needs the ``export`` dependency group (``uv sync --group export``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from iml.data.spectrogram import n_frames
from iml.models import build_model
from iml.runs import load_checkpoint


def export_onnx(run: str | Path, out: str | Path | None = None, opset: int = 17) -> Path:
    path = Path(run)
    ckpt_path = path if path.suffix == ".pt" else path / "best.pt"
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out = Path(out) if out else ckpt_path.parent / "model.onnx"
    dummy = torch.randn(1, 1, cfg.n_mels, n_frames(cfg))
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return out


def check_onnx(onnx_path: str | Path, run: str | Path, atol: float = 1e-3) -> float:
    """Max abs difference between PyTorch and onnxruntime logits on random input (2 lengths)."""
    import onnxruntime as ort

    path = Path(run)
    ckpt = load_checkpoint(path if path.suffix == ".pt" else path / "best.pt")
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for frames in (n_frames(cfg), n_frames(cfg) // 2 + 1):
        x = torch.randn(3, 1, cfg.n_mels, frames)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(["logits"], {"spectrogram": x.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    if worst > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {worst:.2e} (> {atol})")
    return worst


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="export a run's best model to ONNX and verify it")
    p.add_argument("--run", required=True, help="run folder or .pt file")
    p.add_argument("--out", help="output path (default: <run>/model.onnx)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args(argv)
    out = export_onnx(args.run, args.out, args.opset)
    print(f"written {out} ({out.stat().st_size / 1e6:.2f} MB)")
    if not args.no_check:
        diff = check_onnx(out, args.run)
        print(f"onnxruntime matches pytorch (max abs diff {diff:.2e})")


if __name__ == "__main__":
    main()
