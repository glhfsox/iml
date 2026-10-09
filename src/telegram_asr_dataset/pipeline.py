from __future__ import annotations

import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .audio import (
    AudioProcessingError,
    extract_video_audio,
    inspect_wav,
    normalize_audio,
    sha256_file,
)
from .config import ProjectPaths, safe_extension
from .metadata import _calculate_dataset_status, write_metadata_csv
from .models import Candidate
from .state import StateStore


def new_collection_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class DatasetProject:
    def __init__(self, root: Path):
        self.paths = ProjectPaths.from_root(root)
        self.paths.create()
        try:
            os.chmod(self.paths.private, 0o700)
            os.chmod(self.paths.state_db.parent, 0o700)
        except OSError:
            pass
        self.store = StateStore(self.paths.state_db)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> DatasetProject:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.paths.root).as_posix()

    def absolute(self, relative_path: str) -> Path:
        resolved = (self.paths.root / relative_path).resolve()
        resolved.relative_to(self.paths.root)
        return resolved

    def begin(
        self,
        candidate: Candidate,
        collection_run_id: str,
        consent_status: str,
        language: str = "ru",
        consent_ref: str = "",
    ):
        return self.store.begin_candidate(
            candidate, collection_run_id, consent_status, language, consent_ref
        )

    def begin_batch(
        self,
        candidates: list[Candidate],
        collection_run_id: str,
        consent_status: str,
        language: str = "ru",
        consent_ref: str = "",
    ):
        return self.store.begin_candidates(
            candidates,
            collection_run_id,
            consent_status,
            language,
            consent_ref,
        )

    def raw_destination(self, candidate: Candidate, sample_id: str) -> Path:
        fallback = ".ogg" if candidate.media_type == "voice" else ".mp4"
        extension = safe_extension(candidate.extension, fallback=fallback)
        directory = (
            self.paths.raw_voice if candidate.media_type == "voice" else self.paths.raw_video
        )
        return directory / f"{sample_id}{extension}"

    def temporary_download_path(self, candidate: Candidate, sample_id: str) -> Path:
        fallback = ".ogg" if candidate.media_type == "voice" else ".mp4"
        extension = safe_extension(candidate.extension, fallback=fallback)
        return self.paths.downloads / f"{sample_id}.part{extension}"

    def import_local_file(
        self,
        candidate: Candidate,
        source_file: Path,
        collection_run_id: str,
        consent_status: str,
        language: str = "ru",
        consent_ref: str = "",
        move: bool = False,
    ):
        row = self.begin(candidate, collection_run_id, consent_status, language, consent_ref)
        sample_id = str(row["sample_id"])
        if row["processing_status"] == "purged":
            self.export_metadata()
            return self.store.get(sample_id)
        if row["processing_status"] in {"indexed", "excluded"}:
            normalized = str(row["normalized_audio_path"])
            if normalized and self.absolute(normalized).is_file():
                self.export_metadata()
                return self.store.get(sample_id)

        source_file = source_file.resolve()
        if not source_file.is_file() or source_file.stat().st_size == 0:
            self.mark_error(sample_id, "source_missing", "Source media is missing or empty")
            return self.store.get(sample_id)

        raw_destination = self.raw_destination(candidate, sample_id)
        raw_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw_destination.with_name(
            raw_destination.stem + ".part" + raw_destination.suffix
        )
        try:
            if source_file != raw_destination.resolve():
                temporary.unlink(missing_ok=True)
                if move:
                    os.replace(source_file, temporary)
                else:
                    shutil.copyfile(source_file, temporary)
                os.replace(temporary, raw_destination)

            self.store.update(
                sample_id,
                raw_path=self.relative(raw_destination),
                raw_size_bytes=raw_destination.stat().st_size,
                raw_sha256=sha256_file(raw_destination),
                processing_status="downloaded",
                error_code="",
                error="",
            )
            try:
                os.chmod(raw_destination, 0o600)
            except OSError:
                pass
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            self.mark_error(sample_id, "raw_copy_failed", str(exc))
            return self.store.get(sample_id)
        self.export_metadata()
        return self.process_raw(sample_id)

    def process_raw(self, sample_id: str):
        row = self.store.get(sample_id)
        if row is None:
            raise KeyError(sample_id)
        try:
            raw_path = self.absolute(str(row["raw_path"]))
            if not raw_path.is_file():
                raise AudioProcessingError("Saved raw media is missing")

            extracted_path_text = ""
            extracted_size = 0
            source_for_normalization = raw_path
            if row["media_type"] == "video_note":
                extracted_path = self.paths.extracted / f"{sample_id}.mka"
                if not extracted_path.is_file() or extracted_path.stat().st_size == 0:
                    extract_video_audio(raw_path, extracted_path)
                extracted_path_text = self.relative(extracted_path)
                extracted_size = extracted_path.stat().st_size
                source_for_normalization = extracted_path
                self.store.update(
                    sample_id,
                    extracted_audio_path=extracted_path_text,
                    extracted_size_bytes=extracted_size,
                    processing_status="extracted",
                )

            normalized_path = self.paths.normalized / f"{sample_id}.wav"
            if not normalized_path.is_file() or normalized_path.stat().st_size == 0:
                normalize_audio(source_for_normalization, normalized_path)
            metrics = inspect_wav(normalized_path)
            self.store.update(
                sample_id,
                extracted_audio_path=extracted_path_text,
                extracted_size_bytes=extracted_size,
                normalized_audio_path=self.relative(normalized_path),
                normalized_size_bytes=normalized_path.stat().st_size,
                audio_pcm_sha256=metrics.pcm_sha256,
                audio_duration_ms=metrics.duration_ms,
                sample_rate_hz=metrics.sample_rate_hz,
                channels=metrics.channels,
                sample_format=f"pcm_s{metrics.sample_width_bits}le",
                qc_status=metrics.qc_status,
                qc_flags="|".join(metrics.qc_flags),
                rms_dbfs=metrics.rms_dbfs,
                peak_dbfs=metrics.peak_dbfs,
                clipping_ratio=metrics.clipping_ratio,
                silence_ratio=metrics.silence_ratio,
                active_audio_ms=metrics.active_audio_ms,
                active_audio_ratio=metrics.active_audio_ratio,
                estimated_snr_db=metrics.estimated_snr_db,
                quality_score=metrics.quality_score,
                quality_decision=metrics.quality_decision,
                quality_reasons="|".join(metrics.quality_reasons),
                processing_status="normalized",
            )

            duplicate = self.store.find_pcm_duplicate(metrics.pcm_sha256, sample_id)
            if duplicate is not None:
                family_id = str(duplicate["dedupe_family_id"] or duplicate["source_recording_id"])
                self.store.update(
                    sample_id,
                    dedupe_status="exact_duplicate",
                    duplicate_of_source_recording_id=str(duplicate["source_recording_id"]),
                    dedupe_family_id=family_id,
                    split_group_id=family_id,
                    split="excluded",
                    dataset_status="excluded",
                    exclusion_reason="exact_duplicate",
                    processing_status="excluded",
                    processed_at_utc=utc_now_text(),
                    error_code="",
                    error="",
                )
            else:
                policy_excluded = str(row["exclusion_reason"]) == "video_notes_disabled"
                quality_rejected = metrics.quality_decision == "reject"
                dataset_status = (
                    "excluded" if policy_excluded or quality_rejected else "pending_review"
                )
                exclusion_reason = (
                    "video_notes_disabled"
                    if policy_excluded
                    else ("automatic_quality_reject" if quality_rejected else "")
                )
                self.store.update(
                    sample_id,
                    dedupe_status="unique",
                    dataset_status=dataset_status,
                    exclusion_reason=exclusion_reason,
                    split="excluded" if dataset_status == "excluded" else "unassigned",
                    processing_status="indexed",
                    processed_at_utc=utc_now_text(),
                    error_code="",
                    error="",
                )
        except Exception as exc:
            error_code = (
                "audio_processing_failed"
                if isinstance(exc, AudioProcessingError)
                else "processing_failed"
            )
            self.mark_error(sample_id, error_code, str(exc))
        self.export_metadata()
        return self.store.get(sample_id)

    def mark_error(self, sample_id: str, code: str, message: str) -> None:
        clean_message = " ".join(message.split())[:500]
        self.store.update(
            sample_id,
            dataset_status="error",
            processing_status="error",
            error_code=code,
            error=clean_message,
            processed_at_utc=utc_now_text(),
        )
        self.export_metadata()

    def export_metadata(self) -> None:
        write_metadata_csv(self.store, self.paths.metadata_csv)

    def reassess_quality(self) -> dict[str, int]:
        """Recompute speaker-verification quality without re-encoding audio."""

        counts = {
            "accept": 0,
            "review": 0,
            "reject": 0,
            "voice_accept": 0,
            "voice_review": 0,
            "voice_reject": 0,
            "errors": 0,
        }
        for row in self.store.rows():
            if row["processing_status"] in {"purged", "error"}:
                continue
            normalized_text = str(row["normalized_audio_path"])
            if not normalized_text:
                continue
            normalized = self.absolute(normalized_text)
            try:
                metrics = inspect_wav(normalized)
                self.store.update(
                    str(row["sample_id"]),
                    qc_status=metrics.qc_status,
                    qc_flags="|".join(metrics.qc_flags),
                    rms_dbfs=metrics.rms_dbfs,
                    peak_dbfs=metrics.peak_dbfs,
                    clipping_ratio=metrics.clipping_ratio,
                    silence_ratio=metrics.silence_ratio,
                    active_audio_ms=metrics.active_audio_ms,
                    active_audio_ratio=metrics.active_audio_ratio,
                    estimated_snr_db=metrics.estimated_snr_db,
                    quality_score=metrics.quality_score,
                    quality_decision=metrics.quality_decision,
                    quality_reasons="|".join(metrics.quality_reasons),
                )
                updated = self.store.get(str(row["sample_id"]))
                assert updated is not None
                status, reason = _calculate_dataset_status(updated, {})
                split = "excluded" if status == "excluded" else "unassigned"
                self.store.update(
                    str(row["sample_id"]),
                    dataset_status=status,
                    exclusion_reason=reason,
                    split=split,
                    split_version="",
                )
                counts[metrics.quality_decision] += 1
                if row["media_type"] == "voice":
                    counts[f"voice_{metrics.quality_decision}"] += 1
            except Exception as exc:
                counts["errors"] += 1
                self.mark_error(str(row["sample_id"]), "quality_reassessment_failed", str(exc))
        self.export_metadata()
        return counts

    def set_video_notes_policy(self, enabled: bool) -> int:
        """Include or exclude already collected video notes without deleting files."""

        updates: list[tuple[str, dict[str, str]]] = []
        for row in self.store.rows():
            if row["media_type"] != "video_note" or row["processing_status"] == "purged":
                continue
            flags = [flag for flag in str(row["review_flags"]).split("|") if flag]
            if enabled:
                if row["exclusion_reason"] != "video_notes_disabled":
                    continue
                flags = [flag for flag in flags if flag != "video_note_excluded"]
                status, reason = _calculate_dataset_status(row, {"exclusion_reason": ""})
                updates.append(
                    (
                        str(row["sample_id"]),
                        {
                            "dataset_status": status,
                            "exclusion_reason": reason,
                            "split": "unassigned" if status != "excluded" else "excluded",
                            "split_version": "",
                            "review_flags": "|".join(flags),
                        },
                    )
                )
            else:
                if "video_note_excluded" not in flags:
                    flags.append("video_note_excluded")
                updates.append(
                    (
                        str(row["sample_id"]),
                        {
                            "dataset_status": "excluded",
                            "exclusion_reason": "video_notes_disabled",
                            "split": "excluded",
                            "split_version": "",
                            "review_flags": "|".join(flags),
                        },
                    )
                )
        self.store.update_many(updates)
        self.export_metadata()
        return len(updates)

    def purge(self, sample_id: str) -> tuple[list[Path], int]:
        """Delete one sample's media after an explicit consent-revocation action."""

        row = self.store.get(sample_id)
        if row is None:
            raise ValueError(f"Unknown sample_id: {sample_id}")
        pcm_hash = str(row["audio_pcm_sha256"])
        related_duplicates = sum(
            1
            for other in self.store.rows()
            if other["sample_id"] != sample_id
            and pcm_hash
            and str(other["audio_pcm_sha256"]) == pcm_hash
            and other["processing_status"] != "purged"
        )
        removed: list[Path] = []
        for field in ("raw_path", "extracted_audio_path", "normalized_audio_path"):
            relative_path = str(row[field])
            if not relative_path:
                continue
            target = self.absolute(relative_path)
            try:
                target.relative_to(self.paths.data)
            except ValueError as exc:
                raise ValueError("Refusing to purge a path outside data/") from exc
            if target.is_file():
                target.unlink()
                removed.append(target)
            temporary = target.with_name(target.stem + ".part" + target.suffix)
            temporary.unlink(missing_ok=True)
        for temporary in self.paths.downloads.glob(f"{sample_id}.part*"):
            if temporary.is_file():
                temporary.unlink()
                removed.append(temporary)

        self.store.update(
            sample_id,
            sender_ref="",
            raw_path="",
            extracted_audio_path="",
            normalized_audio_path="",
            raw_size_bytes=0,
            extracted_size_bytes=0,
            normalized_size_bytes=0,
            raw_sha256="",
            audio_pcm_sha256="",
            audio_duration_ms=None,
            sample_rate_hz=None,
            channels=None,
            sample_format="",
            transcript="",
            transcript_normalized="",
            transcript_source="none",
            transcript_status="missing",
            target_word="",
            speaker_ref="",
            speaker_status="unknown",
            consent_status="revoked",
            consent_ref="",
            manual_review_status="rejected",
            review_flags="consent_revoked",
            qc_status="pending",
            qc_flags="",
            rms_dbfs=None,
            peak_dbfs=None,
            clipping_ratio=None,
            silence_ratio=None,
            split="excluded",
            split_version="",
            dataset_status="excluded",
            exclusion_reason="consent_revoked_and_purged",
            processing_status="purged",
            error_code="",
            error="",
            processed_at_utc=utc_now_text(),
        )
        self.export_metadata()
        return removed, related_duplicates
