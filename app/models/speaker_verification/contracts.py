"""Data contracts shared by every speaker-verification model pipeline."""

from __future__ import annotations

from dataclasses import dataclass

CLASS_NAMES = {0: "not allowed", 1: "allowed"}


@dataclass(frozen=True)
class Verdict:
    """Decision for one recording."""

    label: int  # 1 = allowed (class 1), 0 = not allowed (class 0)
    confidence: float  # P(predicted class)
    p_allowed: float  # P(class 1) before thresholding
    uncertainty: float  # MC-dropout std of p_allowed (0 when MC dropout is off)

    @property
    def name(self) -> str:
        return CLASS_NAMES[self.label]
