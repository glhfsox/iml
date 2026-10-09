from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from telegram_asr_dataset.models import Candidate
from telegram_asr_dataset.state import StateStore
from telegram_asr_dataset.trials import build_verification_trials


class TrialTests(unittest.TestCase):
    def test_builds_balanced_genuine_and_impostor_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            try:
                message_id = 0
                for split in ("validation", "test"):
                    for speaker in ("speaker_01", "speaker_02"):
                        for session in ("session_a", "session_b"):
                            message_id += 1
                            row = store.begin_candidate(
                                Candidate(
                                    source_type="telegram_export",
                                    source_chat_id="chat",
                                    source_message_id=str(message_id),
                                    message_date_utc=datetime(2026, 10, 6, tzinfo=UTC),
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
                                processing_status="indexed",
                                dedupe_status="unique",
                                speaker_ref=speaker,
                                speaker_status="verified_single",
                                session_ref=session,
                                split=split,
                                normalized_audio_path=f"data/normalized/{message_id}.wav",
                            )
                destination = root / "trials.csv"
                counts = build_verification_trials(store, destination)
                self.assertEqual(counts, {"validation": 4, "test": 4})
                with destination.open(encoding="utf-8") as source:
                    rows = list(csv.DictReader(source))
                for split in ("validation", "test"):
                    labels = [row["expected_same_speaker"] for row in rows if row["split"] == split]
                    self.assertEqual(labels.count("1"), labels.count("0"))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
