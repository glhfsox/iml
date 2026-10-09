from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from telegram_asr_dataset.models import Candidate
from telegram_asr_dataset.split import assert_group_safe, assign_splits
from telegram_asr_dataset.state import StateStore


class SplitTests(unittest.TestCase):
    def test_same_source_group_never_crosses_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            try:
                rows = []
                for index in range(6):
                    row = store.begin_candidate(
                        Candidate(
                            source_type="telegram_export",
                            source_chat_id="chat",
                            source_message_id=str(index),
                            message_date_utc=datetime(2026, 10, 6, index, tzinfo=UTC),
                            media_type="voice",
                            sender_id=None,
                            extension=".ogg",
                        ),
                        "run",
                        "confirmed_other",
                    )
                    store.update(
                        row["sample_id"],
                        dataset_status="ready",
                        dedupe_status="unique",
                        processing_status="indexed",
                        speaker_ref="speaker_01",
                        speaker_status="verified_single",
                        session_ref=f"session_{index}",
                    )
                    rows.append(store.get(row["sample_id"]))
                shared_group = "group_same_original"
                store.update(rows[0]["sample_id"], split_group_id=shared_group)
                store.update(rows[1]["sample_id"], split_group_id=shared_group)
                counts = assign_splits(store, seed=7)
                assert_group_safe(store)
                first = store.get(rows[0]["sample_id"])
                second = store.get(rows[1]["sample_id"])
                self.assertEqual(first["split"], second["split"])
                self.assertEqual(sum(counts.values()), 6)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
