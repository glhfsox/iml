from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .state import StateStore

METADATA_FIELDS = [
    "schema_version",
    "sample_id",
    "source_recording_id",
    "collection_run_id",
    "source_type",
    "chat_ref",
    "message_ref",
    "message_date_utc",
    "media_type",
    "sender_ref",
    "forwarded",
    "raw_path",
    "extracted_audio_path",
    "normalized_audio_path",
    "raw_size_bytes",
    "extracted_size_bytes",
    "normalized_size_bytes",
    "raw_sha256",
    "audio_pcm_sha256",
    "telegram_duration_ms",
    "audio_duration_ms",
    "sample_rate_hz",
    "channels",
    "sample_format",
    "processing_profile",
    "task_type",
    "language",
    "speaker_ref",
    "speaker_status",
    "access_role",
    "session_ref",
    "phrase_id",
    "phrase_status",
    "consent_status",
    "consent_ref",
    "manual_review_status",
    "review_flags",
    "qc_status",
    "qc_flags",
    "rms_dbfs",
    "peak_dbfs",
    "clipping_ratio",
    "silence_ratio",
    "active_audio_ms",
    "active_audio_ratio",
    "estimated_snr_db",
    "quality_score",
    "quality_decision",
    "quality_reasons",
    "dedupe_status",
    "duplicate_of_source_recording_id",
    "dedupe_family_id",
    "recording_group_id",
    "split_group_id",
    "split",
    "split_version",
    "dataset_status",
    "exclusion_reason",
    "processing_status",
    "error_code",
    "processed_at_utc",
]

LABEL_CONTEXT_FIELDS = [
    "normalized_audio_path",
    "message_date_utc",
    "media_type",
    "chat_ref",
    "sender_ref_hint",
    "audio_duration_ms",
    "quality_decision",
    "quality_score",
    "quality_reasons",
]

LABEL_EDIT_FIELDS = [
    "speaker_ref",
    "speaker_status",
    "access_role",
    "session_ref",
    "phrase_id",
    "phrase_status",
    "consent_status",
    "consent_ref",
    "manual_review_status",
    "review_flags",
]

LABEL_FIELDS = ["sample_id", *LABEL_CONTEXT_FIELDS, *LABEL_EDIT_FIELDS]

TRANSCRIPT_STATUSES = {"missing", "draft", "verified"}
SPEAKER_STATUSES = {
    "unreviewed",
    "verified_single",
    "verified_multiple",
    "unknown",
}
CONSENT_STATUSES = {"confirmed_self", "confirmed_other", "unknown", "revoked"}
REVIEW_STATUSES = {"pending", "approved", "rejected"}
ACCESS_ROLES = {"unassigned", "target", "impostor"}
PHRASE_STATUSES = {"not_checked", "correct", "incorrect"}


def _load_speaker_aliases(
    aliases_path: Path | None,
) -> tuple[dict[tuple[str, str], str], dict[str, str], dict[str, str]]:
    """Load optional human-friendly aliases used only as review hints."""
    exact: dict[tuple[str, str], str] = {}
    sender_defaults: dict[str, str] = {}
    chat_defaults: dict[str, str] = {}
    if aliases_path is None or not aliases_path.exists():
        return exact, sender_defaults, chat_defaults

    with aliases_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"chat_ref", "sender_ref", "speaker_ref"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Speaker aliases CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            chat_ref = row["chat_ref"].strip()
            sender_ref = row["sender_ref"].strip()
            speaker_ref = row["speaker_ref"].strip()
            if not speaker_ref:
                raise ValueError(f"Line {line_number}: speaker_ref cannot be empty")
            if chat_ref in {"", "*"} and sender_ref not in {"", "*"}:
                sender_defaults[sender_ref] = speaker_ref
            elif sender_ref in {"", "*"} and chat_ref not in {"", "*"}:
                chat_defaults[chat_ref] = speaker_ref
            elif chat_ref not in {"", "*"} and sender_ref not in {"", "*"}:
                exact[(chat_ref, sender_ref)] = speaker_ref
            else:
                raise ValueError(f"Line {line_number}: set a sender_ref, a chat_ref, or both")
    return exact, sender_defaults, chat_defaults


def _speaker_hint(
    database_row: Any,
    exact: dict[tuple[str, str], str],
    sender_defaults: dict[str, str],
    chat_defaults: dict[str, str],
) -> str:
    # A forwarded message may contain a third person's voice. Its sender is not
    # a safe identity hint, so leave the value blank for explicit review.
    if bool(database_row["forwarded"]):
        return ""
    chat_ref = str(database_row["chat_ref"])
    sender_ref = str(database_row["sender_ref"])
    return (
        exact.get((chat_ref, sender_ref))
        or sender_defaults.get(sender_ref)
        or chat_defaults.get(chat_ref)
        or sender_ref
    )


def normalize_transcript(text: str) -> str:
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=path.name + ".",
        suffix=".part",
        dir=path.parent,
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def write_metadata_csv(store: StateStore, destination: Path) -> None:
    rows = []
    for database_row in store.rows():
        if database_row["processing_status"] == "purged":
            continue
        row = {field: database_row[field] for field in METADATA_FIELDS}
        rows.append(row)
    _write_csv_atomic(destination, METADATA_FIELDS, rows)


def write_labels_template(
    store: StateStore,
    destination: Path,
    collection_run_id: str | None = None,
    speaker_aliases_path: Path | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    exact_aliases, sender_aliases, chat_aliases = _load_speaker_aliases(speaker_aliases_path)
    for database_row in store.rows(collection_run_id):
        if database_row["processing_status"] in {"purged", "error"}:
            continue
        if database_row["dataset_status"] == "excluded":
            continue
        values = {field: database_row[field] for field in LABEL_EDIT_FIELDS}
        # The account sender is only a candidate identity. It stays unreviewed until a human
        # listens to the audio and explicitly approves the row.
        if not values["speaker_ref"] and database_row["sender_ref"]:
            values["speaker_ref"] = _speaker_hint(
                database_row, exact_aliases, sender_aliases, chat_aliases
            )
        if not values["session_ref"] and database_row["message_date_utc"]:
            recording_day = str(database_row["message_date_utc"])[:10]
            values["session_ref"] = f"day_{recording_day}"
        context = {
            field: (
                database_row["sender_ref"] if field == "sender_ref_hint" else database_row[field]
            )
            for field in LABEL_CONTEXT_FIELDS
        }
        rows.append({"sample_id": database_row["sample_id"], **context, **values})
    _write_csv_atomic(destination, LABEL_FIELDS, rows)


def _calculate_dataset_status(row: Any, values: dict[str, str]) -> tuple[str, str]:
    dedupe_status = str(row["dedupe_status"])
    qc_status = str(row["qc_status"])
    processing_status = str(row["processing_status"])
    consent = values.get("consent_status", str(row["consent_status"]))
    consent_ref = values.get("consent_ref", str(row["consent_ref"]))
    review = values.get("manual_review_status", str(row["manual_review_status"]))
    speaker_status = values.get("speaker_status", str(row["speaker_status"]))
    speaker_ref = values.get("speaker_ref", str(row["speaker_ref"]))
    session_ref = values.get("session_ref", str(row["session_ref"]))
    quality_decision = str(row["quality_decision"])

    policy_reason = values.get("exclusion_reason", str(row["exclusion_reason"]))
    if policy_reason == "video_notes_disabled":
        return "excluded", "video_notes_disabled"
    if dedupe_status == "exact_duplicate":
        return "excluded", "exact_duplicate"
    if consent == "revoked":
        return "excluded", "consent_revoked"
    if review == "rejected":
        return "excluded", "manual_review_rejected"
    if qc_status == "fail" or quality_decision == "reject":
        return "excluded", "automatic_quality_reject"
    ready = (
        processing_status == "indexed"
        and dedupe_status == "unique"
        and (consent == "confirmed_self" or (consent == "confirmed_other" and bool(consent_ref)))
        and review == "approved"
        and qc_status in {"pass", "warn"}
        and quality_decision in {"accept", "review"}
        and speaker_status == "verified_single"
        and bool(speaker_ref)
        and bool(session_ref)
    )
    return ("ready", "") if ready else ("pending_review", "")


def apply_labels_csv(store: StateStore, labels_path: Path) -> int:
    pending_updates: list[tuple[str, dict[str, str]]] = []
    seen_sample_ids: set[str] = set()
    with labels_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(LABEL_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels CSV is missing columns: {sorted(missing)}")
        for line_number, label_row in enumerate(reader, start=2):
            sample_id = label_row["sample_id"].strip()
            if sample_id in seen_sample_ids:
                raise ValueError(f"Line {line_number}: duplicate sample_id {sample_id!r}")
            seen_sample_ids.add(sample_id)
            database_row = store.get(sample_id)
            if database_row is None:
                raise ValueError(f"Line {line_number}: unknown sample_id {sample_id!r}")
            values = {field: label_row[field].strip() for field in LABEL_EDIT_FIELDS}
            if values["speaker_status"] not in SPEAKER_STATUSES:
                raise ValueError(
                    f"Line {line_number}: invalid speaker_status {values['speaker_status']!r}"
                )
            if values["consent_status"] not in CONSENT_STATUSES:
                raise ValueError(
                    f"Line {line_number}: invalid consent_status {values['consent_status']!r}"
                )
            if values["manual_review_status"] not in REVIEW_STATUSES:
                raise ValueError(
                    f"Line {line_number}: invalid manual_review_status "
                    f"{values['manual_review_status']!r}"
                )
            if values["access_role"] not in ACCESS_ROLES:
                raise ValueError(
                    f"Line {line_number}: invalid access_role {values['access_role']!r}"
                )
            if values["phrase_status"] not in PHRASE_STATUSES:
                raise ValueError(
                    f"Line {line_number}: invalid phrase_status {values['phrase_status']!r}"
                )
            if (
                values["speaker_status"] in {"verified_single", "verified_multiple"}
                and not values["speaker_ref"]
            ):
                raise ValueError(
                    f"Line {line_number}: verified speaker status requires speaker_ref"
                )
            if values["manual_review_status"] == "approved":
                if values["speaker_status"] != "verified_single":
                    raise ValueError(
                        f"Line {line_number}: approved audio requires exactly one verified speaker"
                    )
                if not values["session_ref"]:
                    raise ValueError(f"Line {line_number}: approved audio requires session_ref")
            if values["speaker_ref"] and values["session_ref"]:
                values["split_group_id"] = (
                    f"session:{values['speaker_ref']}:{values['session_ref']}"
                )
            dataset_status, exclusion_reason = _calculate_dataset_status(database_row, values)
            values["dataset_status"] = dataset_status
            values["exclusion_reason"] = exclusion_reason
            if dataset_status == "excluded":
                values["split"] = "excluded"
                values["split_version"] = ""
            elif dataset_status == "pending_review":
                values["split"] = "unassigned"
                values["split_version"] = ""
            elif str(database_row["split"]) == "excluded":
                values["split"] = "unassigned"
                values["split_version"] = ""
            pending_updates.append((sample_id, values))
    store.update_many(pending_updates)
    return len(pending_updates)
