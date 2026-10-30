"""Input contract of WeSpeaker ResNet34-LM: waveform -> fbank features, per-record segmentation.

Input : mono float waveform in [-1, 1] at ``SAMPLE_RATE`` (16 kHz) Hz, any length.
Tensor: per segment a ``[T, 80]`` float32 log-mel fbank with CMN (see :func:`fbank`).
A record is split into segments (``PreprocCfg.mode``: full / energy-VAD / sliding chunks); every
segment is embedded and the L2-normalised embeddings are averaged (:func:`embed_record`). The
preprocessing a bank was built with is stored in its ``meta["preproc"]`` and reused at inference.

Fbank matches WeSpeaker's ONNX inference recipe (``wespeaker/bin/infer_onnx.py``):
int16-scaled wave, 80 mel bins, 25/10 ms hamming frames, no dither, no energy, CMN without CVN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from app.models.utils import Operator

if TYPE_CHECKING:
    from .embedder import WeSpeakerEmbedder

SAMPLE_RATE = 16000
MODES = ("full", "vad", "chunks")

Wave = torch.Tensor | np.ndarray


@dataclass
class PreprocCfg:
    mode: str = "full"  # "full" | "vad" | "chunks"
    win_s: float = 3.0  # chunks mode
    hop_s: float = 1.5
    vad_top_db: float = 35.0  # vad: keep frames within top_db of the loudest frame
    vad_frame_ms: float = 30.0
    min_keep_s: float = 1.0  # vad falls back to the full wave if less speech is kept

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocCfg:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _to_tensor(wave: Wave) -> torch.Tensor:
    return torch.as_tensor(wave).detach().float().reshape(-1)


def fbank(wave: Wave) -> np.ndarray:
    """1D wave in [-1, 1] @ 16 kHz -> ``[T, 80]`` float32 log-mel fbank with CMN."""
    w = _to_tensor(wave)
    min_len = int(0.025 * SAMPLE_RATE)
    if w.numel() < min_len:  # kaldi.fbank needs at least one frame
        w = torch.nn.functional.pad(w, (0, min_len - w.numel()))
    feats = kaldi.fbank(
        w[None] * (1 << 15),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=SAMPLE_RATE,
        window_type="hamming",
        use_energy=False,
    )
    feats = feats - feats.mean(dim=0)
    return feats.numpy().astype(np.float32, copy=False)


def energy_vad(
    wave: Wave,
    top_db: float = 35.0,
    frame_ms: float = 30.0,
    min_keep_s: float = 1.0,
    sr: int = SAMPLE_RATE,
) -> Wave:
    """Drop frames more than ``top_db`` below the loudest frame; returns the same type as input.

    Falls back to the untouched wave when less than ``min_keep_s`` of audio would remain.
    """
    is_tensor = isinstance(wave, torch.Tensor)
    x = wave.detach().cpu().numpy() if is_tensor else np.asarray(wave)
    x = x.reshape(-1)
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = len(x) // frame
    if n_frames == 0:
        return wave
    framed = x[: n_frames * frame].reshape(n_frames, frame).astype(np.float64)
    db = 10.0 * np.log10(np.mean(framed**2, axis=1) + 1e-12)
    keep = db >= db.max() - top_db
    kept = framed[keep].reshape(-1)
    # keep the partial tail frame if its last full neighbour was kept
    if keep[-1] and len(x) > n_frames * frame:
        kept = np.concatenate([kept, x[n_frames * frame :].astype(np.float64)])
    if len(kept) < min_keep_s * sr:
        return wave
    out = kept.astype(x.dtype, copy=False)
    return torch.from_numpy(np.ascontiguousarray(out)) if is_tensor else out


def chunks(wave: Wave, win_s: float = 3.0, hop_s: float = 1.5, sr: int = SAMPLE_RATE) -> list:
    """Sliding windows of ``win_s`` every ``hop_s``; a final window is aligned to the end."""
    n = wave.shape[-1]
    win, hop = int(win_s * sr), max(1, int(hop_s * sr))
    if n <= win:
        return [wave]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)  # tail chunk covering the end
    return [wave[s : s + win] for s in starts]


def segments(wave: Wave, cfg: PreprocCfg) -> list:
    """Split one record into the segments that get embedded and averaged."""
    if cfg.mode == "full":
        return [wave]
    if cfg.mode == "vad":
        return [energy_vad(wave, cfg.vad_top_db, cfg.vad_frame_ms, cfg.min_keep_s)]
    if cfg.mode == "chunks":
        return chunks(wave, cfg.win_s, cfg.hop_s)
    raise ValueError(f"unknown mode {cfg.mode!r}")


def _l2(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def record_feats(wave: Wave, cfg: PreprocCfg) -> list[np.ndarray]:
    """One ``[T, 80]`` fbank per segment of the record."""
    return [fbank(seg) for seg in segments(wave, cfg)]


def pool(embs: list[np.ndarray]) -> np.ndarray:
    """Segment embeddings -> L2-normalise each, average, renormalise -> ``[D]`` float32."""
    return _l2(np.mean([_l2(e) for e in embs], axis=0)).astype(np.float32)


def embed_record(embedder: WeSpeakerEmbedder, wave: Wave, cfg: PreprocCfg) -> np.ndarray:
    """Embed each segment, L2-normalise, average, renormalise -> ``[D]`` float32."""
    return pool([embedder.embed_feats(f) for f in record_feats(wave, cfg)])


class FbankTransform(Operator):
    """Mono waveform at 16 kHz -> list of ``[T, 80]`` fbank segments (:func:`record_feats`)."""

    def __init__(self, cfg: PreprocCfg) -> None:
        self.cfg = cfg

    def __call__(self, wave: Wave) -> list[np.ndarray]:
        if wave.ndim != 1:
            raise TypeError("expected a 1-D mono waveform")
        return record_feats(wave, self.cfg)


def make_transform(cfg: PreprocCfg) -> FbankTransform:
    return FbankTransform(cfg)
