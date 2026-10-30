"""Pipeline of WeSpeaker ResNet34-LM (pretrained, zero-shot): build_pipeline(...)(wave) -> Verdict.

Unlike the trained folders there is no ``config.json``/``best.pt``: the ONNX graph is pretrained
and the speaker-specific part is an embedding bank (``.npz``) passed by path.
"""

from __future__ import annotations

from pathlib import Path

from app.models.speaker_verification.wespeaker_resnet34.embedder import WeSpeakerEmbedder
from app.models.speaker_verification.wespeaker_resnet34.pipeline.operators import (
    EmbeddingBank,
    operators,
)
from app.models.speaker_verification.wespeaker_resnet34.transform import SAMPLE_RATE
from app.models.utils import Pipeline

__all__ = ["SAMPLE_RATE", "EmbeddingBank", "build_pipeline"]


def build_pipeline(
    bank: str | Path | EmbeddingBank,
    threshold: float | None = None,
    embedder: WeSpeakerEmbedder | None = None,
) -> Pipeline:
    """``bank``: path to an embedding bank ``.npz``; ``threshold=None`` uses the bank's own."""
    bank = bank if isinstance(bank, EmbeddingBank) else EmbeddingBank.load(bank)
    thr = bank.threshold if threshold is None else threshold
    return Pipeline(*operators(bank, embedder or WeSpeakerEmbedder(), thr))
