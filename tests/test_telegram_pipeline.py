from __future__ import annotations

import csv
import math
import struct
import subprocess
import tempfile
import unittest
import wave
from datetime import UTC, datetime
from pathlib import Path

from telegram_asr_dataset.audio import AudioProcessingError, find_ffmpeg
from telegram_asr_dataset.models import Candidate
from telegram_asr_dataset.pipeline import DatasetProject


def write_tone(path: Path, frequency: float = 440.0) -> None:
    sample_rate = 8_000
    frames = bytearray()
    for index in range(sample_rate):
        value = int(8_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))


def candidate(message_id: str, media_type: str = "voice", extension: str = ".wav") -> Candidate:
    return Candidate(
        source_type="telegram_export",
        source_chat_id="private-real-chat-id",
        source_message_id=message_id,
        message_date_utc=datetime(2026, 10, 6, 12, tzinfo=UTC),
        media_type=media_type,
        sender_id="private-real-sender-id",
        extension=extension,
    )


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            find_ffmpeg()
        except AudioProcessingError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_normalizes_deduplicates_and_hides_raw_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tone_a = root / "a.wav"
            tone_b = root / "b.wav"
            write_tone(tone_a)
            write_tone(tone_b)
            with DatasetProject(root) as project:
                first = project.import_local_file(
                    candidate("100"), tone_a, "pilot", "confirmed_other"
                )
                second = project.import_local_file(
                    candidate("101"), tone_b, "pilot", "confirmed_other"
                )
                assert first is not None and second is not None
                self.assertEqual(first["sample_rate_hz"], 16_000)
                self.assertEqual(first["channels"], 1)
                self.assertEqual(first["dedupe_status"], "unique")
                self.assertEqual(second["dedupe_status"], "exact_duplicate")
                self.assertEqual(
                    project.absolute(first["raw_path"]).stat().st_mode & 0o777,
                    0o600,
                )
                self.assertEqual(project.paths.raw_voice.stat().st_mode & 0o777, 0o700)
                self.assertEqual(project.paths.data.stat().st_mode & 0o777, 0o700)
                metadata_text = project.paths.metadata_csv.read_text(encoding="utf-8")
                self.assertNotIn("private-real-chat-id", metadata_text)
                self.assertNotIn("private-real-sender-id", metadata_text)
                with project.paths.metadata_csv.open(encoding="utf-8") as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), 2)

                # Repairing the canonical row must not make it a duplicate of its
                # already-excluded duplicate.
                project.absolute(first["normalized_audio_path"]).unlink()
                repaired = project.process_raw(first["sample_id"])
                self.assertEqual(repaired["dedupe_status"], "unique")
                self.assertEqual(repaired["processing_status"], "indexed")

    def test_purge_removes_media_but_keeps_reimport_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tone = root / "source.wav"
            write_tone(tone)
            with DatasetProject(root) as project:
                row = project.import_local_file(
                    candidate("300"),
                    tone,
                    "pilot",
                    "confirmed_other",
                    consent_ref="consent_batch_01",
                )
                raw = project.absolute(row["raw_path"])
                normalized = project.absolute(row["normalized_audio_path"])
                removed, _ = project.purge(row["sample_id"])
                self.assertIn(raw, removed)
                self.assertFalse(raw.exists())
                self.assertFalse(normalized.exists())
                tombstone = project.store.get(row["sample_id"])
                self.assertEqual(tombstone["processing_status"], "purged")
                self.assertEqual(tombstone["consent_status"], "revoked")
                metadata = project.paths.metadata_csv.read_text(encoding="utf-8")
                self.assertNotIn(row["sample_id"], metadata)

    def test_video_note_keeps_raw_and_extracts_separate_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "circle.mp4"
            subprocess.run(
                [
                    find_ffmpeg(),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=64x64:d=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=500:duration=1",
                    "-shortest",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "aac",
                    str(video),
                ],
                check=True,
            )
            with DatasetProject(root) as project:
                row = project.import_local_file(
                    candidate("200", "video_note", ".mp4"),
                    video,
                    "pilot",
                    "confirmed_other",
                )
                assert row is not None
                self.assertTrue(project.absolute(row["raw_path"]).is_file())
                self.assertTrue(project.absolute(row["extracted_audio_path"]).is_file())
                self.assertTrue(project.absolute(row["normalized_audio_path"]).is_file())

                changed = project.set_video_notes_policy(False)
                excluded = project.store.get(row["sample_id"])
                self.assertEqual(changed, 1)
                self.assertEqual(excluded["dataset_status"], "excluded")
                self.assertEqual(excluded["exclusion_reason"], "video_notes_disabled")
                self.assertTrue(project.absolute(excluded["raw_path"]).is_file())

                project.set_video_notes_policy(True)
                restored = project.store.get(row["sample_id"])
                self.assertEqual(restored["dataset_status"], "excluded")
                self.assertEqual(restored["exclusion_reason"], "automatic_quality_reject")


if __name__ == "__main__":
    unittest.main()
