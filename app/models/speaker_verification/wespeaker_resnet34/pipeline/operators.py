"""Operator chain of WeSpeaker ResNet34-LM: fbank -> ONNX embedding -> bank cosine -> verdict.

Zero-shot: the model is never trained on our speakers. Enrollment is an embedding bank, a ``.npz``
saved by ``Gallery.save`` (``notebooks/glhfsox/speaker_id/gallery.py``): L2-normalised speaker
``centroids`` + ``speakers`` + JSON ``meta`` with the decision ``threshold`` and the ``preproc``
the bank was embedded with. A recording is allowed if its best centroid cosine >= threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from app.models.speaker_verification.wespeaker_resnet34.embedder import WeSpeakerEmbedder
from app.models.speaker_verification.wespeaker_resnet34.transform import (
    PreprocCfg,
    make_transform,
    pool,
)
from app.models.utils import DecodeVerdict, Operator


@dataclass(frozen=True)
class EmbeddingBank:
    """Speaker centroids + calibration of an embedding bank (the zero-shot "weights")."""

    centroids: np.ndarray  # [S, D], L2-normalised
    speakers: list[str]
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingBank:
        with np.load(path, allow_pickle=False) as z:
            centroids = np.asarray(z["centroids"], dtype=np.float32)
            speakers = [str(s) for s in z["speakers"].tolist()]
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if "threshold" not in meta:
            raise ValueError(f"{path} has no meta['threshold']; not a calibrated bank")
        # this pipeline scores raw cosine to centroids; refuse banks calibrated otherwise
        mode, norm = meta.get("mode", "centroid"), meta.get("norm", "raw")
        if (mode, norm) != ("centroid", "raw"):
            raise ValueError(
                f"{path}: bank calibrated with mode={mode!r} norm={norm!r}; "
                "only centroid/raw scoring is supported here"
            )
        return cls(centroids, speakers, meta, Path(path))

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def preproc(self) -> PreprocCfg:
        return PreprocCfg.from_dict(self.meta.get("preproc", {}))


class Embed(Operator):
    """List of ``[T, 80]`` fbank segments -> pooled L2-normalised ``[D]`` embedding."""

    def __init__(self, embedder: WeSpeakerEmbedder) -> None:
        self.embedder = embedder

    def __call__(self, feats: list[np.ndarray]) -> np.ndarray:
        return pool([self.embedder.embed_feats(f) for f in feats])


class BankScore(Operator):
    """Embedding -> ``(score, std)`` for a batch of one, the input :class:`DecodeVerdict` takes.

    ``score`` is the best cosine to a bank centroid clipped to [0, 1]: a similarity, not a
    calibrated probability; the bank threshold is what separates allowed from not allowed.
    """

    def __init__(self, bank: EmbeddingBank) -> None:
        self.bank = bank

    def __call__(self, emb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.bank.centroids @ np.asarray(emb, dtype=np.float32)
        best = float(np.clip(cos.max(), 0.0, 1.0))
        return torch.tensor([best]), torch.zeros(1)


def operators(bank: EmbeddingBank, embedder: WeSpeakerEmbedder, threshold: float) -> list[Operator]:
    ckpt = bank.meta.get("checkpoint")
    if ckpt and ckpt != embedder.name:
        raise ValueError(f"embedder {embedder.name!r} != bank checkpoint {ckpt!r}")
    return [
        make_transform(bank.preproc),
        Embed(embedder),
        BankScore(bank),
        DecodeVerdict(threshold),
    ]
