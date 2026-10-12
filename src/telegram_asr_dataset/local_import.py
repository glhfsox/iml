from __future__ import annotations

import json
import re
import shutil
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .audio import inspect_wav, normalize_audio, sha256_file
from .metadata import _calculate_dataset_status, write_labels_template
from .models import Candidate
from .pipeline import DatasetProject, new_collection_run_id

SUPPORTED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return result or "recording"


def _chunk_pcm16_wav(
    source: Path, destination: Path, chunk_seconds: float, minimum_seconds: float = 2.0
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16_000 or wav.getsampwidth() != 2:
            raise ValueError("local import expects normalized mono 16 kHz PCM16 WAV")
        total_frames = wav.getnframes()
        chunk_frames = round(chunk_seconds * 16_000)
        minimum_frames = round(minimum_seconds * 16_000)
        boundaries: list[tuple[int, int]] = []
        start = 0
        while total_frames - start > chunk_frames:
            boundaries.append((start, start + chunk_frames))
            start += chunk_frames
        remaining = total_frames - start
        if remaining >= minimum_frames or not boundaries:
            boundaries.append((start, total_frames))
        elif remaining > 0:
            previous_start, _ = boundaries[-1]
            boundaries[-1] = (previous_start, total_frames)

        chunks: list[Path] = []
        for index, (first, last) in enumerate(boundaries, start=1):
            wav.setpos(first)
            target = destination / f"chunk_{index:04d}.wav"
            with wave.open(str(target), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(wav.readframes(last - first))
            chunks.append(target)
        return chunks


def import_local_speaker(
    project_root: Path,
    input_dir: Path,
    speaker_ref: str,
    language: str,
    consent_ref: str,
    chunk_seconds: float = 20.0,
) -> dict[str, object]:
    project_root = project_root.resolve()
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if not speaker_ref.strip() or not consent_ref.strip():
        raise ValueError("speaker_ref and consent_ref are required")
    if not 5.0 <= chunk_seconds <= 45.0:
        raise ValueError("chunk_seconds must be between 5 and 45")

    sources = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    if not sources:
        raise ValueError("no supported audio files found")

    run_id = new_collection_run_id()
    workspace = project_root / "work" / "local_import" / run_id
    archive = project_root / "data" / "raw" / "local_sources" / speaker_ref
    workspace.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    imported_review_ids: list[int] = []

    with DatasetProject(project_root) as project:
        try:
            for source in sources:
                source_hash = sha256_file(source)
                source_slug = _slug(source.stem)
                archived = archive / f"{source_hash[:12]}_{source.name}"
                if not archived.exists():
                    shutil.copy2(source, archived)
                normalized_source = workspace / f"{source_hash[:16]}.wav"
                normalize_audio(source, normalized_source)
                chunks = _chunk_pcm16_wav(
                    normalized_source, workspace / source_hash[:16], chunk_seconds
                )
                source_group = f"local:{speaker_ref}:{source_hash[:16]}"
                session_ref = f"local_{speaker_ref.casefold()}_{source_slug}"
                timestamp = datetime.fromtimestamp(source.stat().st_mtime, tz=UTC)
                for index, chunk in enumerate(chunks, start=1):
                    metrics = inspect_wav(chunk)
                    candidate = Candidate(
                        source_type="local_directory",
                        source_chat_id=f"local:{speaker_ref}",
                        source_message_id=f"{source_hash}:chunk:{index:04d}",
                        message_date_utc=timestamp,
                        media_type="voice",
                        sender_id=f"local:{speaker_ref}",
                        extension=".wav",
                        telegram_duration_ms=metrics.duration_ms,
                    )
                    row = project.import_local_file(
                        candidate,
                        chunk,
                        run_id,
                        consent_status="confirmed_other",
                        language=language,
                        consent_ref=consent_ref,
                    )
                    if row is None:
                        counts["error"] += 1
                        continue
                    sample_id = str(row["sample_id"])
                    values = {
                        "speaker_ref": speaker_ref,
                        "speaker_status": "verified_single",
                        "access_role": "target",
                        "session_ref": session_ref,
                        "phrase_id": source_slug,
                        "phrase_status": "not_checked",
                        "manual_review_status": "approved",
                        "review_flags": "direct_team_recording",
                        "recording_group_id": source_group,
                        "split_group_id": f"session:{speaker_ref}:{session_ref}",
                    }
                    status, reason = _calculate_dataset_status(row, values)
                    values.update(
                        dataset_status=status,
                        exclusion_reason=reason,
                        split="excluded" if status == "excluded" else "unassigned",
                        split_version="",
                    )
                    project.store.update(sample_id, **values)
                    updated = project.store.get(sample_id)
                    assert updated is not None
                    counts[str(updated["dataset_status"])] += 1
                    imported_review_ids.append(int(updated["review_id"]))
            project.store.mark_run_complete(run_id)
            project.export_metadata()
            labels_path = project.paths.labels / f"speaker_labels_{run_id}.csv"
            write_labels_template(project.store, labels_path, run_id)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    report = {
        "run_id": run_id,
        "speaker_ref": speaker_ref,
        "language": language,
        "source_files": len(sources),
        "chunks": len(imported_review_ids),
        "chunk_seconds": chunk_seconds,
        "counts": dict(counts),
        "review_ids": imported_review_ids,
        "labels": str(labels_path),
        "originals_archive": str(archive),
        "note": "Chunks from one source file share one split group.",
    }
    report_path = project_root / "data" / "reports" / f"local_import_{run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
