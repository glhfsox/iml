from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass(frozen=True)
class VerificationDecision:
    decision: str
    median_similarity: float
    minimum_similarity: float
    threshold: float


def classify_similarities(scores: list[float], threshold: float) -> VerificationDecision:
    if not scores:
        return VerificationDecision("insufficient_speech", math.nan, math.nan, threshold)
    score_median = float(median(scores))
    score_minimum = float(min(scores))
    if score_median >= threshold and score_minimum >= threshold - 0.08:
        decision = "auto_likely_match"
    elif score_median < threshold - 0.12:
        decision = "auto_likely_mismatch"
    else:
        decision = "uncertain"
    return VerificationDecision(decision, score_median, score_minimum, threshold)


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _audio_path(root: Path, row: dict[str, str]) -> Path:
    value = row.get("normalized_audio_path", "").strip()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"audio path escapes project root: {value!r}") from exc
    return candidate


def _speech_windows(path: Path, seconds: float = 3.0, maximum: int = 5) -> list[torch.Tensor]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16_000:
        raise ValueError(f"expected 16 kHz WAV, got {sample_rate}: {path}")
    mono = audio.mean(axis=1)
    window = round(seconds * sample_rate)
    if len(mono) < sample_rate:
        return []
    starts = list(range(0, max(1, len(mono) - window + 1), window))
    candidates: list[tuple[float, np.ndarray]] = []
    for start in starts:
        chunk = mono[start : start + window]
        if len(chunk) < sample_rate:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        dbfs = 20.0 * math.log10(max(rms, 1e-8))
        if dbfs >= -45.0:
            candidates.append((rms, chunk))
    if not candidates:
        return []
    # Keep diverse positions, then restore time order. This bounds runtime for long messages.
    indices = np.linspace(0, len(candidates) - 1, min(maximum, len(candidates)), dtype=int)
    return [torch.from_numpy(candidates[index][1]) for index in sorted(set(indices.tolist()))]


def _recording_embeddings(model: Any, path: Path) -> list[torch.Tensor]:
    windows = _speech_windows(path)
    if not windows:
        return []
    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for waveform in windows:
            value = model.encode_batch(waveform.unsqueeze(0)).squeeze().float().cpu()
            embeddings.append(F.normalize(value, dim=0))
    return embeddings


def _mean_embedding(embeddings: list[torch.Tensor]) -> torch.Tensor:
    return F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item())


def verify_collection_run(
    metadata_csv: Path,
    project_root: Path,
    run_id: str,
    claimed_speaker: str,
    output_dir: Path,
) -> dict[str, Any]:
    rows = _read_metadata(metadata_csv)
    references = [
        row
        for row in rows
        if row.get("speaker_ref") == claimed_speaker
        and row.get("speaker_status") == "verified_single"
        and row.get("manual_review_status") == "approved"
        and row.get("forwarded") in {"0", "False", "false", ""}
    ]
    if len(references) < 3:
        raise ValueError(
            f"speaker {claimed_speaker!r} needs at least 3 manually approved references"
        )
    candidates = [
        row
        for row in rows
        if row.get("collection_run_id") == run_id
        and row.get("forwarded") in {"0", "False", "false", ""}
        and row.get("normalized_audio_path")
    ]
    if not candidates:
        raise ValueError(f"no non-forwarded audio found for run {run_id!r}")

    try:
        os.environ.setdefault(
            "HF_HOME", str(project_root / "data" / "private" / "huggingface-cache")
        )
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise RuntimeError("install the data dependencies: uv sync --group data") from exc

    model_dir = project_root / "data" / "private" / "models" / "spkrec-ecapa-voxceleb"
    model = EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=str(model_dir))

    reference_embeddings: list[torch.Tensor] = []
    for row in references:
        embeddings = _recording_embeddings(model, _audio_path(project_root, row))
        if embeddings:
            reference_embeddings.append(_mean_embedding(embeddings))
    if len(reference_embeddings) < 3:
        raise ValueError("fewer than 3 reference recordings contain enough audible speech")

    centroid = F.normalize(torch.stack(reference_embeddings).mean(dim=0), dim=0)
    positive_scores = [_cosine(value, centroid) for value in reference_embeddings]
    negative_rows = [
        row
        for row in rows
        if row.get("speaker_ref") not in {"", claimed_speaker}
        and row.get("speaker_status") == "verified_single"
        and row.get("manual_review_status") == "approved"
        and row.get("forwarded") in {"0", "False", "false", ""}
    ]
    negative_scores: list[float] = []
    for row in negative_rows:
        embeddings = _recording_embeddings(model, _audio_path(project_root, row))
        if embeddings:
            negative_scores.append(_cosine(_mean_embedding(embeddings), centroid))
    negative_ceiling = max(negative_scores, default=0.25)
    # Conservative local calibration: observed impostors plus a margin, never below 0.35.
    threshold = min(0.85, max(0.35, negative_ceiling + 0.05))

    output_rows: list[dict[str, Any]] = []
    for row in candidates:
        path = _audio_path(project_root, row)
        embeddings = _recording_embeddings(model, path)
        scores = [_cosine(embedding, centroid) for embedding in embeddings]
        decision = classify_similarities(scores, threshold)
        output_rows.append(
            {
                "review_id": row.get("review_id", ""),
                "sample_id": row.get("sample_id", ""),
                "collection_run_id": run_id,
                "claimed_speaker": claimed_speaker,
                "normalized_audio_path": row.get("normalized_audio_path", ""),
                "speech_windows": len(embeddings),
                "cosine_similarity": (
                    ""
                    if math.isnan(decision.median_similarity)
                    else f"{decision.median_similarity:.6f}"
                ),
                "minimum_window_similarity": (
                    ""
                    if math.isnan(decision.minimum_similarity)
                    else f"{decision.minimum_similarity:.6f}"
                ),
                "threshold": f"{threshold:.6f}",
                "decision": decision.decision,
                "verification_method": "ecapa_tdnn_weak_label",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"auto_verification_{run_id}_{claimed_speaker}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    counts: dict[str, int] = {}
    for row in output_rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL_SOURCE,
        "run_id": run_id,
        "claimed_speaker": claimed_speaker,
        "manual_reference_recordings": len(reference_embeddings),
        "known_impostor_recordings": len(negative_scores),
        "positive_similarity_min": min(positive_scores),
        "known_impostor_similarity_max": max(negative_scores, default=None),
        "threshold": threshold,
        "counts": counts,
        "csv": str(csv_path),
        "warning": "Automatic verification is a weak label, not proof of identity.",
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
