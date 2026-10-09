from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from telegram_asr_dataset.cli import (
    _ensure_selected_chat_coverage,
    _ordered_items,
    _select_collection_batch,
)
from telegram_asr_dataset.models import Candidate
from telegram_asr_dataset.pipeline import DatasetProject


@dataclass(frozen=True)
class Item:
    candidate: Candidate


@dataclass(frozen=True)
class Chat:
    source_chat_id: str
    title: str


def make_items(count: int, chat_count: int = 1) -> list[Item]:
    base = datetime(2026, 10, 1, tzinfo=UTC)
    return [
        Item(
            Candidate(
                source_type="telegram_export",
                source_chat_id=f"chat_{index % chat_count}",
                source_message_id=str(index),
                message_date_utc=base + timedelta(minutes=index),
                media_type="voice",
                sender_id=None,
                extension=".ogg",
            )
        )
        for index in range(count)
    ]


class CollectionBatchTests(unittest.TestCase):
    def test_every_selected_chat_must_contribute_candidates(self) -> None:
        items = make_items(4, chat_count=2)
        selected = [Chat("chat_0", "A"), Chat("chat_1", "B")]
        _ensure_selected_chat_coverage(selected, items)
        with self.assertRaisesRegex(ValueError, "C"):
            _ensure_selected_chat_coverage([*selected, Chat("chat_2", "C")], items)

    def test_interrupted_pilot_resumes_the_same_fixed_twenty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            items = make_items(30)
            with DatasetProject(Path(directory)) as project:
                run_id, window, pending, resuming = _select_collection_batch(
                    project, items, 20, "oldest", "global", False, False
                )
                self.assertFalse(resuming)
                self.assertEqual(len(window), 20)
                project.begin_batch(
                    [item.candidate for item in window],
                    run_id,
                    "confirmed_other",
                    consent_ref="consent_batch_01",
                )
                for item in pending[:5]:
                    row = project.store.get_by_source(item.candidate)
                    project.store.update(
                        row["sample_id"],
                        processing_status="indexed",
                        dedupe_status="unique",
                    )

                resumed_id, resumed_window, resumed_pending, is_resuming = _select_collection_batch(
                    project, items, 20, "oldest", "global", False, False
                )
                self.assertTrue(is_resuming)
                self.assertEqual(resumed_id, run_id)
                self.assertEqual(len(resumed_window), 20)
                self.assertEqual(len(resumed_pending), 15)
                self.assertEqual(
                    {item.candidate.source_message_id for item in resumed_window},
                    {str(index) for index in range(20)},
                )

    def test_next_batch_starts_after_completed_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            items = make_items(30)
            with DatasetProject(Path(directory)) as project:
                run_id, window, _, _ = _select_collection_batch(
                    project, items, 20, "oldest", "global", False, False
                )
                rows = project.begin_batch(
                    [item.candidate for item in window],
                    run_id,
                    "confirmed_other",
                    consent_ref="consent_batch_01",
                )
                project.store.update_many(
                    [
                        (
                            row["sample_id"],
                            {"processing_status": "indexed", "dedupe_status": "unique"},
                        )
                        for row in rows
                    ]
                )
                project.store.mark_run_complete(run_id)
                _, next_window, next_pending, resuming = _select_collection_batch(
                    project, items, 20, "oldest", "global", True, False
                )
                self.assertFalse(resuming)
                self.assertEqual(len(next_window), 10)
                self.assertEqual(len(next_pending), 10)
                self.assertEqual(
                    {item.candidate.source_message_id for item in next_window},
                    {str(index) for index in range(20, 30)},
                )

    def test_balanced_sampling_covers_selected_chats(self) -> None:
        ordered = _ordered_items(make_items(12, chat_count=3), "newest", "balanced")
        self.assertEqual(
            {item.candidate.source_chat_id for item in ordered[:3]},
            {"chat_0", "chat_1", "chat_2"},
        )

    def test_balanced_sampling_uses_sender_as_unverified_hint(self) -> None:
        base = datetime(2026, 10, 1, tzinfo=UTC)
        items = []
        for index in range(8):
            items.append(
                Item(
                    Candidate(
                        source_type="telegram_export",
                        source_chat_id=f"chat_{index % 2}",
                        source_message_id=str(index),
                        message_date_utc=base + timedelta(minutes=index),
                        media_type="voice",
                        sender_id=f"sender_{(index // 2) % 2}",
                        extension=".ogg",
                    )
                )
            )
        ordered = _ordered_items(items, "oldest", "balanced")
        first_buckets = {
            (item.candidate.source_chat_id, item.candidate.sender_id) for item in ordered[:4]
        }
        self.assertEqual(len(first_buckets), 4)


if __name__ == "__main__":
    unittest.main()
