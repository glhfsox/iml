"""Prepare and load a privacy-preserving speaker-verification dataset manifest.

The Telegram collector is intentionally kept outside the training package.  It produces a
reviewed ``metadata.csv`` plus normalized WAV files.  This module is the boundary between
those private collection artifacts and the model code:

* accept only manually approved, consented, single-speaker recordings;
* derive a binary one-vs-rest label for the selected target speaker;
* keep every speaker session intact across train/val/test;
* copy only the normalized WAV files and a minimal pseudonymous manifest;
* fingerprint the exact split membership for reproducible experiment names.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
REQUIRED_METADATA_FIELDS = {
    "sample_id",
    "source_recording_id",
    "media_type",
    "forwarded",
    "normalized_audio_path",
    "audio_pcm_sha256",
    "speaker_ref",
    "speaker_status",
    "session_ref",
    "consent_status",
    "manual_review_status",
    "quality_decision",
    "dedupe_status",
    "dataset_status",
}


@dataclass(frozen=True)
class SourceRecording:
    sample_id: str
    source_recording_id: str
    source_path: Path
    speaker_ref: str
    session_ref: str
    pcm_sha256: str
    duration_s: float


@dataclass(frozen=True)
class ManifestItem:
    sample_id: str
    source_recording_id: str
    path: str
    label: int
    speaker_ref: str
    session_ref: str
    split: str
    pcm_sha256: str
    duration_s: float


def _pcm_sha256(path: Path) -> str:
    """Hash decoded PCM frames exactly as the collector does, excluding WAV headers."""

    digest = hashlib.sha256()
    try:
        with wave.open(str(path), "rb") as source:
            for frames in iter(lambda: source.readframes(65_536), b""):
                digest.update(frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"{path.name}: cannot hash PCM frames: {exc}") from exc
    return digest.hexdigest()


def _safe_source_path(source_root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (source_root / candidate).resolve()
    )
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"audio path escapes source root: {value!r}") from exc
    if not resolved.is_file():
        raise ValueError(f"normalized audio does not exist: {value!r}")
    return resolved


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.format != "WAV" or info.samplerate != 16_000 or info.channels != 1:
        raise ValueError(
            f"{path.name}: expected mono 16 kHz WAV, got "
            f"{info.format} {info.samplerate} Hz {info.channels} channels"
        )
    if info.subtype != "PCM_16":
        raise ValueError(f"{path.name}: expected PCM_16 WAV, got {info.subtype}")
    duration = info.frames / info.samplerate
    if not 2.0 <= duration <= 45.0:
        raise ValueError(f"{path.name}: duration {duration:.2f}s is outside 2-45 seconds")
    return duration


def _exclusion_reason(row: dict[str, str]) -> str | None:
    if row["media_type"] != "voice":
        return "not_voice"
    if row["forwarded"].strip().lower() in {"1", "true", "yes"}:
        return "forwarded"
    if row["dataset_status"] == "excluded":
        return row.get("exclusion_reason", "").strip() or "excluded"
    if row["manual_review_status"] != "approved":
        if row["manual_review_status"] == "pending":
            return "pending_manual_review"
        return "manual_review_not_approved"
    if row["speaker_status"] != "verified_single":
        return "speaker_not_verified_single"
    if row["consent_status"] not in {"confirmed_self", "confirmed_other"}:
        return "consent_not_confirmed"
    if row["quality_decision"] not in {"accept", "review"}:
        return "quality_rejected"
    if row["dedupe_status"] != "unique":
        return "duplicate"
    if not row["speaker_ref"].strip():
        return "missing_speaker_ref"
    if not row["session_ref"].strip():
        return "missing_session_ref"
    if not row["normalized_audio_path"].strip():
        return "missing_audio_path"
    if row["dataset_status"] != "ready":
        return "not_ready"
    return None


def read_reviewed_metadata(
    metadata_csv: str | Path, source_root: str | Path
) -> tuple[list[SourceRecording], Counter[str]]:
    """Read collector metadata and return only rows safe for supervised training."""

    metadata_csv = Path(metadata_csv).resolve()
    source_root = Path(source_root).resolve()
    exclusions: Counter[str] = Counter()
    recordings: list[SourceRecording] = []
    with metadata_csv.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_METADATA_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            reason = _exclusion_reason(row)
            if reason:
                exclusions[reason] += 1
                continue
            try:
                audio_path = _safe_source_path(source_root, row["normalized_audio_path"])
                duration_s = _audio_duration(audio_path)
            except ValueError as exc:
                raise ValueError(f"metadata line {line_number}: {exc}") from exc
            actual_pcm_sha256 = _pcm_sha256(audio_path)
            pcm_sha256 = row["audio_pcm_sha256"].strip() or actual_pcm_sha256
            if pcm_sha256 != actual_pcm_sha256:
                raise ValueError(
                    f"metadata line {line_number}: PCM hash does not match {audio_path.name}"
                )
            recordings.append(
                SourceRecording(
                    sample_id=row["sample_id"].strip(),
                    source_recording_id=row["source_recording_id"].strip()
                    or row["sample_id"].strip(),
                    source_path=audio_path,
                    speaker_ref=row["speaker_ref"].strip(),
                    session_ref=row["session_ref"].strip(),
                    pcm_sha256=pcm_sha256,
                    duration_s=duration_s,
                )
            )

    if not recordings:
        raise ValueError("metadata contains no reviewed recordings ready for training")
    sample_counts = Counter(row.sample_id for row in recordings)
    duplicate_samples = sorted(sample for sample, count in sample_counts.items() if count > 1)
    if duplicate_samples:
        raise ValueError(f"duplicate sample_id values: {duplicate_samples[:5]}")
    hashes: dict[str, str] = {}
    for row in recordings:
        prior = hashes.setdefault(row.pcm_sha256, row.sample_id)
        if prior != row.sample_id:
            raise ValueError(
                f"duplicate PCM audio in reviewed rows: {prior!r} and {row.sample_id!r}"
            )
    return recordings, exclusions


def _stable_order(values: list[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest(),
    )


def _assign_groups(
    group_weights: dict[str, int], seed: int, namespace: str, ratios: dict[str, float]
) -> dict[str, str]:
    if len(group_weights) < len(SPLITS):
        raise ValueError(f"{namespace} needs at least three independent groups")
    ordered = _stable_order(list(group_weights), seed, namespace)
    assignments = {group: split for group, split in zip(ordered[:3], SPLITS, strict=True)}
    current = {split: group_weights[group] for group, split in assignments.items()}
    total_weight = sum(group_weights.values())
    for group in ordered[3:]:
        split = max(
            SPLITS,
            key=lambda name: (ratios[name] * total_weight - current[name], -SPLITS.index(name)),
        )
        assignments[group] = split
        current[split] += group_weights[group]
    return assignments


def assign_speaker_verification_splits(
    recordings: list[SourceRecording],
    target_speaker: str,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Assign sample IDs to splits by whole sessions, stratified per speaker.

    With the four-person team dataset, every speaker contributes independent sessions to
    train, validation and test.  This gives the model examples of all three impostors while
    ensuring that recordings from one session never leak across splits.
    """

    ratios = dict(ratios or DEFAULT_RATIOS)
    if set(ratios) != set(SPLITS) or any(value <= 0 for value in ratios.values()):
        raise ValueError("ratios must contain positive train, val and test values")
    total_ratio = sum(ratios.values())
    ratios = {key: value / total_ratio for key, value in ratios.items()}
    speakers = {row.speaker_ref for row in recordings}
    if target_speaker not in speakers:
        raise ValueError(f"target speaker {target_speaker!r} is absent from reviewed metadata")
    if len(speakers) < 2:
        raise ValueError("speaker verification needs a target and at least one impostor speaker")

    session_assignments: dict[str, dict[str, str]] = {}
    for speaker in sorted(speakers):
        session_counts: Counter[str] = Counter(
            row.session_ref for row in recordings if row.speaker_ref == speaker
        )
        session_assignments[speaker] = _assign_groups(
            dict(session_counts), seed, f"speaker:{speaker}", ratios
        )

    assignments: dict[str, str] = {}
    for row in recordings:
        assignments[row.sample_id] = session_assignments[row.speaker_ref][row.session_ref]
    return assignments


def _dataset_id(items: list[ManifestItem], target_speaker: str) -> str:
    stable = {
        "target_speaker": target_speaker,
        "items": [
            {
                "sample_id": item.sample_id,
                "source_recording_id": item.source_recording_id,
                "label": item.label,
                "speaker_ref": item.speaker_ref,
                "session_ref": item.session_ref,
                "split": item.split,
                "pcm_sha256": item.pcm_sha256,
            }
            for item in sorted(items, key=lambda value: value.sample_id)
        ],
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _copy_audio(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return
    if copy_mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return
    raise ValueError("copy_mode must be copy or hardlink")


def _validate_items(items: list[ManifestItem]) -> None:
    labels_by_split: dict[str, set[int]] = defaultdict(set)
    source_splits: dict[str, str] = {}
    session_splits: dict[tuple[str, str], str] = {}
    for item in items:
        if item.split not in SPLITS or item.label not in {0, 1}:
            raise ValueError(f"invalid split/label for {item.sample_id}")
        labels_by_split[item.split].add(item.label)
        prior = source_splits.setdefault(item.source_recording_id, item.split)
        if prior != item.split:
            raise ValueError(f"source recording leakage: {item.source_recording_id}")
        session_key = (item.speaker_ref, item.session_ref)
        prior = session_splits.setdefault(session_key, item.split)
        if prior != item.split:
            raise ValueError(f"speaker session leakage: {item.speaker_ref}/{item.session_ref}")
    for split in SPLITS:
        if labels_by_split[split] != {0, 1}:
            raise ValueError(f"split {split!r} must contain target and impostor recordings")


def prepare_dataset(
    metadata_csv: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    target_speaker: str,
    *,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    copy_mode: str = "copy",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a minimal local training dataset from reviewed collector metadata."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    recordings, exclusions = read_reviewed_metadata(metadata_csv, source_root)
    assignments = assign_speaker_verification_splits(
        recordings,
        target_speaker,
        seed=seed,
        ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
    )

    items = [
        ManifestItem(
            sample_id=row.sample_id,
            source_recording_id=row.source_recording_id,
            path=f"audio/{row.sample_id}.wav",
            label=int(row.speaker_ref == target_speaker),
            speaker_ref=row.speaker_ref,
            session_ref=row.session_ref,
            split=assignments[row.sample_id],
            pcm_sha256=row.pcm_sha256,
            duration_s=round(row.duration_s, 3),
        )
        for row in recordings
    ]
    _validate_items(items)
    dataset_id = _dataset_id(items, target_speaker)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        by_sample = {row.sample_id: row for row in recordings}
        for item in items:
            _copy_audio(
                by_sample[item.sample_id].source_path,
                staging / item.path,
                copy_mode,
            )

        fieldnames = ["dataset_id", *ManifestItem.__dataclass_fields__]
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for item in sorted(items, key=lambda value: (value.split, value.sample_id)):
                writer.writerow({"dataset_id": dataset_id, **asdict(item)})

        split_payload = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "target_speaker": target_speaker,
            "splits": {
                split: [
                    asdict(item)
                    for item in sorted(items, key=lambda value: value.sample_id)
                    if item.split == split
                ]
                for split in SPLITS
            },
        }
        (staging / "splits.json").write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        counts = {
            split: {
                "total": sum(item.split == split for item in items),
                "target": sum(item.split == split and item.label == 1 for item in items),
                "impostor": sum(item.split == split and item.label == 0 for item in items),
            }
            for split in SPLITS
        }
        report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset_id": dataset_id,
            "target_speaker": target_speaker,
            "approved_recordings": len(items),
            "total_duration_s": round(sum(item.duration_s for item in items), 3),
            "speakers": sorted({item.speaker_ref for item in items}),
            "target_sessions": sorted({item.session_ref for item in items if item.label == 1}),
            "counts": counts,
            "excluded_rows": dict(sorted(exclusions.items())),
            "split_policy": {
                "all_speakers": "whole (speaker_ref, session_ref) groups",
                "stratification": "each speaker contributes sessions to every split",
                "seed": seed,
            },
        }
        (staging / "dataset_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(staging, output_dir)
    return report


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Validate ``splits.json`` and resolve its audio paths for training."""

    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("dataset_id"):
        raise ValueError("unsupported or incomplete split manifest")
    if set(payload.get("splits", {})) != set(SPLITS):
        raise ValueError("split manifest must contain train, val and test")

    items: list[ManifestItem] = []
    seen_samples: set[str] = set()
    base = path.parent
    for split in SPLITS:
        for raw in payload["splits"][split]:
            raw = dict(raw)
            raw["split"] = split
            item = ManifestItem(**raw)
            if item.sample_id in seen_samples:
                raise ValueError(f"duplicate sample_id in split manifest: {item.sample_id}")
            seen_samples.add(item.sample_id)
            resolved = (base / item.path).resolve()
            try:
                resolved.relative_to(base)
            except ValueError as exc:
                raise ValueError(f"manifest audio path escapes dataset: {item.path!r}") from exc
            if not resolved.is_file():
                raise ValueError(f"manifest audio is missing: {item.path!r}")
            _audio_duration(resolved)
            if _pcm_sha256(resolved) != item.pcm_sha256:
                raise ValueError(f"manifest audio hash mismatch: {item.path!r}")
            items.append(item)
    _validate_items(items)
    target_speaker = payload.get("target_speaker")
    if not isinstance(target_speaker, str) or not target_speaker:
        raise ValueError("split manifest is missing target_speaker")
    if _dataset_id(items, target_speaker) != payload["dataset_id"]:
        raise ValueError("split manifest fingerprint does not match its contents")
    payload["resolved_items"] = {
        split: [((base / item.path).resolve(), item.label) for item in items if item.split == split]
        for split in SPLITS
    }
    return payload


def copy_manifest_to_run(manifest_path: str | Path, run_dir: str | Path) -> Path:
    """Record the exact split membership beside model metrics and checkpoints."""

    source = Path(manifest_path).resolve()
    destination = Path(run_dir) / "splits.json"
    shutil.copy2(source, destination)
    return destination
