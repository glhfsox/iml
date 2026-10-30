"""WeSpeaker ResNet34-LM (VoxCeleb2, 256-d) speaker embedder on onnxruntime.

Pretrained, not trained by us: there is no ``best.pt``/``config.json``. The ONNX graph lives next
to this file (gitignored) and is downloaded from Hugging Face on first use
(:func:`ensure_checkpoint`).
The file keeps its upstream name because embedding banks record it in ``meta["checkpoint"]``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

from iml.data import load_wav

from .transform import SAMPLE_RATE, PreprocCfg, Wave, embed_record

CHECKPOINT_NAME = "voxceleb_resnet34_LM.onnx"
HF_REPO = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM"
CHECKPOINT_URL = f"{HF_REPO}/resolve/main/{CHECKPOINT_NAME}"

DEFAULT_DIR = Path(__file__).resolve().parent


def ensure_checkpoint(dest_dir: Path | None = None) -> Path:
    """Return the local ONNX path, downloading it (atomically via ``.part``) if missing."""
    dest_dir = Path(dest_dir) if dest_dir is not None else DEFAULT_DIR
    path = dest_dir / CHECKPOINT_NAME
    if path.exists():
        return path
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(CHECKPOINT_URL) as r, open(part, "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def _n_threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class WeSpeakerEmbedder:
    """``[T, 80]`` fbank -> L2-normalised speaker embedding via an ONNX session."""

    def __init__(self, path: Path | str | None = None, providers: list[str] | None = None):
        path = Path(path) if path is not None else ensure_checkpoint()
        so = ort.SessionOptions()
        # explicit thread count avoids pthread_setaffinity errors in containers
        so.intra_op_num_threads = _n_threads()
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=so, providers=providers or ["CPUExecutionProvider"]
        )
        self.name = path.name
        self.input_name = self.session.get_inputs()[0].name
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        self.dim = int(out.shape[-1])

    def embed_feats(self, feats: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(feats, dtype=np.float32)[None]
        emb = self.session.run([self.output_name], {self.input_name: x})[0][0]
        return (emb / max(float(np.linalg.norm(emb)), 1e-12)).astype(np.float32)

    def embed_wave(self, wave: Wave, cfg: PreprocCfg | None = None) -> np.ndarray:
        return embed_record(self, wave, cfg or PreprocCfg())

    def embed_file(self, path: str | Path, cfg: PreprocCfg | None = None) -> np.ndarray:
        return self.embed_wave(load_wav(path, SAMPLE_RATE), cfg)
