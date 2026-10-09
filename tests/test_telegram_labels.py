from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from telegram_asr_dataset.metadata import LABEL_FIELDS, apply_labels_csv, write_labels_template
from telegram_asr_dataset.models import Candidate
from telegram_asr_dataset.state import StateStore


def make_ready_for_labels(store: StateStore, message_id: str):
    row = store.begin_candidate(
        Candidate(
            source_type="telegram_export",
            source_chat_id="chat",
            source_message_id=message_id,
            message_date_utc=datetime(2026, 10, 6, tzinfo=UTC),
            media_type="voice",
            sender_id="sender",
            extension=".ogg",
        ),
        "run",
        "confirmed_other",
        consent_ref="consent_batch_01",
    )
    store.update(
        row["sample_id"],
        processing_status="indexed",
        dedupe_status="unique",
        qc_status="pass",
        quality_decision="accept",
    )
    return store.get(row["sample_id"])


def label_row(sample_id: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "speaker_ref": "speaker_01",
        "speaker_status": "verified_single",
        "access_role": "target",
        "session_ref": f"session_{sample_id}",
        "phrase_id": "",
        "phrase_status": "not_checked",
        "consent_status": "confirmed_other",
        "consent_ref": "consent_batch_01",
        "manual_review_status": "approved",
        "review_flags": "",
    }


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class LabelTests(unittest.TestCase):
    def test_review_template_prefills_unconfirmed_sender_and_day_hints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                row = make_ready_for_labels(store, "hint")
                destination = root / "labels.csv"
                write_labels_template(store, destination)
                with destination.open(encoding="utf-8") as source:
                    label = next(csv.DictReader(source))
                self.assertEqual(label["sender_ref_hint"], row["sender_ref"])
                self.assertEqual(label["speaker_ref"], row["sender_ref"])
                self.assertEqual(label["session_ref"], "day_2026-10-06")
                self.assertEqual(label["speaker_status"], "unreviewed")
                self.assertEqual(label["manual_review_status"], "pending")
            finally:
                store.close()

    def test_private_aliases_prefill_short_codes_without_verifying_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                self_row = make_ready_for_labels(store, "self")
                other = store.begin_candidate(
                    Candidate(
                        source_type="telegram_export",
                        source_chat_id="chat",
                        source_message_id="other",
                        message_date_utc=datetime(2026, 10, 7, tzinfo=UTC),
                        media_type="voice",
                        sender_id="other_sender",
                        extension=".ogg",
                    ),
                    "run",
                    "confirmed_other",
                    consent_ref="consent_batch_01",
                )
                store.update(
                    other["sample_id"],
                    processing_status="indexed",
                    dedupe_status="unique",
                    qc_status="pass",
                    quality_decision="accept",
                )
                other = store.get(other["sample_id"])
                aliases = root / "speaker_aliases.csv"
                with aliases.open("w", encoding="utf-8", newline="") as target:
                    writer = csv.DictWriter(
                        target, fieldnames=["chat_ref", "sender_ref", "speaker_ref"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "chat_ref": "*",
                            "sender_ref": self_row["sender_ref"],
                            "speaker_ref": "P",
                        }
                    )
                    writer.writerow(
                        {
                            "chat_ref": other["chat_ref"],
                            "sender_ref": "*",
                            "speaker_ref": "T",
                        }
                    )
                destination = root / "labels.csv"
                write_labels_template(store, destination, speaker_aliases_path=aliases)
                with destination.open(encoding="utf-8") as source:
                    labels = {row["sample_id"]: row for row in csv.DictReader(source)}
                self.assertEqual(labels[self_row["sample_id"]]["speaker_ref"], "P")
                self.assertEqual(labels[other["sample_id"]]["speaker_ref"], "T")
                self.assertEqual(labels[other["sample_id"]]["speaker_status"], "unreviewed")
            finally:
                store.close()

    def test_forwarded_review_template_leaves_speaker_hint_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                row = store.begin_candidate(
                    Candidate(
                        source_type="telegram_export",
                        source_chat_id="chat",
                        source_message_id="forwarded",
                        message_date_utc=datetime(2026, 10, 6, tzinfo=UTC),
                        media_type="voice",
                        sender_id="sender",
                        extension=".ogg",
                        forwarded=True,
                    ),
                    "run",
                    "confirmed_other",
                    consent_ref="consent_batch_01",
                )
                store.update(
                    row["sample_id"],
                    processing_status="indexed",
                    dedupe_status="unique",
                    qc_status="pass",
                    quality_decision="accept",
                )
                aliases = root / "speaker_aliases.csv"
                with aliases.open("w", encoding="utf-8", newline="") as target:
                    writer = csv.DictWriter(
                        target, fieldnames=["chat_ref", "sender_ref", "speaker_ref"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "chat_ref": "*",
                            "sender_ref": row["sender_ref"],
                            "speaker_ref": "P",
                        }
                    )
                destination = root / "labels.csv"
                write_labels_template(store, destination, speaker_aliases_path=aliases)
                with destination.open(encoding="utf-8") as source:
                    label = next(csv.DictReader(source))
                self.assertEqual(label["speaker_ref"], "")
                self.assertEqual(label["speaker_status"], "unreviewed")
            finally:
                store.close()

    def test_human_can_approve_borderline_review_but_not_auto_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                review_row = make_ready_for_labels(store, "review")
                reject_row = make_ready_for_labels(store, "reject")
                store.update(
                    review_row["sample_id"],
                    qc_status="warn",
                    quality_decision="review",
                )
                store.update(
                    reject_row["sample_id"],
                    qc_status="fail",
                    quality_decision="reject",
                )
                labels = root / "labels.csv"
                write_labels(
                    labels,
                    [
                        label_row(review_row["sample_id"]),
                        label_row(reject_row["sample_id"]),
                    ],
                )
                apply_labels_csv(store, labels)
                self.assertEqual(store.get(review_row["sample_id"])["dataset_status"], "ready")
                self.assertEqual(store.get(reject_row["sample_id"])["dataset_status"], "excluded")
            finally:
                store.close()

    def test_approved_base_recording_does_not_need_a_permanent_access_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                row = make_ready_for_labels(store, "unassigned-role")
                label = label_row(row["sample_id"])
                label["access_role"] = "unassigned"
                labels = root / "labels.csv"
                write_labels(labels, [label])
                apply_labels_csv(store, labels)
                updated = store.get(row["sample_id"])
                self.assertEqual(updated["dataset_status"], "ready")
                self.assertEqual(updated["access_role"], "unassigned")
            finally:
                store.close()

    def test_invalid_later_row_rolls_back_all_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                first = make_ready_for_labels(store, "1")
                second = make_ready_for_labels(store, "2")
                rows = [label_row(first["sample_id"]), label_row(second["sample_id"])]
                rows[1]["speaker_status"] = "invalid"
                labels = root / "labels.csv"
                write_labels(labels, rows)
                with self.assertRaises(ValueError):
                    apply_labels_csv(store, labels)
                self.assertEqual(store.get(first["sample_id"])["transcript"], "")
            finally:
                store.close()

    def test_revocation_clears_existing_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                row = make_ready_for_labels(store, "1")
                initial = label_row(row["sample_id"])
                labels = root / "labels.csv"
                write_labels(labels, [initial])
                apply_labels_csv(store, labels)
                self.assertEqual(store.get(row["sample_id"])["dataset_status"], "ready")
                store.update(row["sample_id"], split="train", split_version="v1")

                initial["consent_status"] = "revoked"
                initial["consent_ref"] = ""
                write_labels(labels, [initial])
                apply_labels_csv(store, labels)
                updated = store.get(row["sample_id"])
                self.assertEqual(updated["dataset_status"], "excluded")
                self.assertEqual(updated["split"], "excluded")
                self.assertEqual(updated["split_version"], "")
            finally:
                store.close()

    def test_forwarded_message_does_not_inherit_batch_consent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            try:
                row = store.begin_candidate(
                    Candidate(
                        source_type="telegram_export",
                        source_chat_id="chat",
                        source_message_id="1",
                        message_date_utc=datetime(2026, 10, 6, tzinfo=UTC),
                        media_type="voice",
                        sender_id="sender",
                        extension=".ogg",
                        forwarded=True,
                    ),
                    "run",
                    "confirmed_other",
                    consent_ref="consent_batch_01",
                )
                self.assertEqual(row["consent_status"], "unknown")
                self.assertEqual(row["consent_ref"], "")
                self.assertIn("consent_needs_review", row["review_flags"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
