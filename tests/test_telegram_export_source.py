from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from telegram_asr_dataset.export_source import export_items, load_export_chats


class ExportSourceTests(unittest.TestCase):
    def test_filters_media_and_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "voice_messages" / "audio_1.ogg"
            media.parent.mkdir()
            media.write_bytes(b"not-real-audio")
            payload = {
                "name": "Учебный чат",
                "type": "personal_chat",
                "id": 42,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date_unixtime": "1791300000",
                        "from_id": "user1",
                        "media_type": "voice_message",
                        "file": "voice_messages/audio_1.ogg",
                        "duration_seconds": 3,
                        "text": "подпись — не транскрипция",
                    },
                    {
                        "id": 2,
                        "type": "message",
                        "date_unixtime": "1791300001",
                        "from_id": "user2",
                        "media_type": "video_message",
                        "file": "../outside.mp4",
                        "duration_seconds": 5,
                    },
                    {
                        "id": 3,
                        "type": "message",
                        "date_unixtime": "1791300002",
                        "file": "video_files/ordinary.mp4",
                        "media_type": "video_file",
                    },
                ],
            }
            (root / "result.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            chats = load_export_chats(root)
            items = export_items(
                root,
                chats,
                datetime(2026, 10, 1, tzinfo=UTC),
                datetime(2026, 10, 10, tzinfo=UTC),
                include_video_notes=True,
            )
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].candidate.media_type, "voice")
            self.assertEqual(items[0].media_path, media.resolve())
            self.assertIsNone(items[1].media_path)
            self.assertIn("escapes", items[1].path_error)

            voice_only = export_items(
                root,
                chats,
                datetime(2026, 10, 1, tzinfo=UTC),
                datetime(2026, 10, 10, tzinfo=UTC),
            )
            self.assertEqual(len(voice_only), 1)
            self.assertEqual(voice_only[0].candidate.media_type, "voice")


if __name__ == "__main__":
    unittest.main()
