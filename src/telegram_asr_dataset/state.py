from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from collections.abc import Iterable
from datetime import UTC
from pathlib import Path
from typing import Any

from .models import Candidate

SCHEMA_VERSION = "3"


class StateStore:
    """Private canonical state; metadata.csv is regenerated from this database."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS collection_runs (
                run_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'processing'
            );

            CREATE TABLE IF NOT EXISTS recordings (
                sample_id TEXT PRIMARY KEY,
                review_id INTEGER UNIQUE,
                schema_version TEXT NOT NULL,
                collection_run_id TEXT NOT NULL,
                source_recording_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_chat_id TEXT NOT NULL,
                source_message_id TEXT NOT NULL,
                chat_ref TEXT NOT NULL,
                message_ref TEXT NOT NULL,
                message_date_utc TEXT NOT NULL,
                media_type TEXT NOT NULL,
                sender_ref TEXT NOT NULL DEFAULT '',
                forwarded INTEGER NOT NULL DEFAULT 0,
                telegram_duration_ms INTEGER,
                raw_path TEXT NOT NULL DEFAULT '',
                extracted_audio_path TEXT NOT NULL DEFAULT '',
                normalized_audio_path TEXT NOT NULL DEFAULT '',
                raw_size_bytes INTEGER NOT NULL DEFAULT 0,
                extracted_size_bytes INTEGER NOT NULL DEFAULT 0,
                normalized_size_bytes INTEGER NOT NULL DEFAULT 0,
                raw_sha256 TEXT NOT NULL DEFAULT '',
                audio_pcm_sha256 TEXT NOT NULL DEFAULT '',
                audio_duration_ms INTEGER,
                sample_rate_hz INTEGER,
                channels INTEGER,
                sample_format TEXT NOT NULL DEFAULT '',
                processing_profile TEXT NOT NULL DEFAULT 'wav_pcm_s16le_16000_mono_v1',
                task_type TEXT NOT NULL DEFAULT 'speaker_verification',
                language TEXT NOT NULL DEFAULT 'ru',
                transcript TEXT NOT NULL DEFAULT '',
                transcript_normalized TEXT NOT NULL DEFAULT '',
                transcript_source TEXT NOT NULL DEFAULT 'none',
                transcript_status TEXT NOT NULL DEFAULT 'missing',
                target_word TEXT NOT NULL DEFAULT '',
                speaker_ref TEXT NOT NULL DEFAULT '',
                speaker_status TEXT NOT NULL DEFAULT 'unreviewed',
                access_role TEXT NOT NULL DEFAULT 'unassigned',
                session_ref TEXT NOT NULL DEFAULT '',
                phrase_id TEXT NOT NULL DEFAULT '',
                phrase_status TEXT NOT NULL DEFAULT 'not_checked',
                consent_status TEXT NOT NULL DEFAULT 'unknown',
                consent_ref TEXT NOT NULL DEFAULT '',
                manual_review_status TEXT NOT NULL DEFAULT 'pending',
                review_flags TEXT NOT NULL DEFAULT '',
                qc_status TEXT NOT NULL DEFAULT 'pending',
                qc_flags TEXT NOT NULL DEFAULT '',
                rms_dbfs REAL,
                peak_dbfs REAL,
                clipping_ratio REAL,
                silence_ratio REAL,
                active_audio_ms INTEGER,
                active_audio_ratio REAL,
                estimated_snr_db REAL,
                quality_score INTEGER,
                quality_decision TEXT NOT NULL DEFAULT 'pending',
                quality_reasons TEXT NOT NULL DEFAULT '',
                dedupe_status TEXT NOT NULL DEFAULT 'unchecked',
                duplicate_of_source_recording_id TEXT NOT NULL DEFAULT '',
                dedupe_family_id TEXT NOT NULL DEFAULT '',
                recording_group_id TEXT NOT NULL,
                split_group_id TEXT NOT NULL,
                split TEXT NOT NULL DEFAULT 'unassigned',
                split_version TEXT NOT NULL DEFAULT '',
                dataset_status TEXT NOT NULL DEFAULT 'pending_review',
                exclusion_reason TEXT NOT NULL DEFAULT '',
                processing_status TEXT NOT NULL DEFAULT 'discovered',
                error_code TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                processed_at_utc TEXT NOT NULL DEFAULT '',
                UNIQUE(source_type, source_chat_id, source_message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_recordings_run
                ON recordings(collection_run_id);
            CREATE INDEX IF NOT EXISTS idx_recordings_pcm_hash
                ON recordings(audio_pcm_sha256);
            CREATE INDEX IF NOT EXISTS idx_recordings_raw_hash
                ON recordings(raw_sha256);
            CREATE INDEX IF NOT EXISTS idx_recordings_split_group
                ON recordings(split_group_id);

            CREATE TABLE IF NOT EXISTS collection_run_items (
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (run_id, sample_id),
                FOREIGN KEY (run_id) REFERENCES collection_runs(run_id),
                FOREIGN KEY (sample_id) REFERENCES recordings(sample_id)
            );
            """
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO settings(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        run_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(collection_runs)")
        }
        if "status" not in run_columns:
            self.connection.execute(
                "ALTER TABLE collection_runs ADD COLUMN status TEXT NOT NULL DEFAULT 'processing'"
            )
        recording_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(recordings)")
        }
        migrations = {
            "review_id": "INTEGER",
            "task_type": "TEXT NOT NULL DEFAULT 'speaker_verification'",
            "access_role": "TEXT NOT NULL DEFAULT 'unassigned'",
            "session_ref": "TEXT NOT NULL DEFAULT ''",
            "phrase_id": "TEXT NOT NULL DEFAULT ''",
            "phrase_status": "TEXT NOT NULL DEFAULT 'not_checked'",
            "active_audio_ms": "INTEGER",
            "active_audio_ratio": "REAL",
            "estimated_snr_db": "REAL",
            "quality_score": "INTEGER",
            "quality_decision": "TEXT NOT NULL DEFAULT 'pending'",
            "quality_reasons": "TEXT NOT NULL DEFAULT ''",
        }
        for column, definition in migrations.items():
            if column not in recording_columns:
                self.connection.execute(f"ALTER TABLE recordings ADD COLUMN {column} {definition}")
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_recordings_review_id ON recordings(review_id)"
        )
        self._backfill_review_ids()
        self.connection.execute(
            "UPDATE settings SET value = ? WHERE key = 'schema_version'",
            (SCHEMA_VERSION,),
        )
        self.connection.execute(
            "UPDATE recordings SET schema_version = ? WHERE schema_version != ?",
            (SCHEMA_VERSION, SCHEMA_VERSION),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO settings(key, value) VALUES ('privacy_salt', ?)",
            (os.urandom(32).hex(),),
        )
        self.connection.commit()

    def _backfill_review_ids(self) -> None:
        """Assign permanent human-facing numbers without reusing retired records."""
        next_id = int(
            self.connection.execute(
                "SELECT COALESCE(MAX(review_id), 0) + 1 FROM recordings"
            ).fetchone()[0]
        )
        seen: set[str] = set()
        ordered = self.connection.execute(
            """
            SELECT recordings.sample_id
              FROM collection_runs
              JOIN collection_run_items USING (run_id)
              JOIN recordings USING (sample_id)
             WHERE recordings.review_id IS NULL
             ORDER BY collection_runs.rowid, collection_run_items.position, recordings.rowid
            """
        )
        sample_ids = [str(row[0]) for row in ordered]
        sample_ids.extend(
            str(row[0])
            for row in self.connection.execute(
                "SELECT sample_id FROM recordings WHERE review_id IS NULL ORDER BY rowid"
            )
        )
        for sample_id in sample_ids:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            self.connection.execute(
                "UPDATE recordings SET review_id = ? WHERE sample_id = ? AND review_id IS NULL",
                (next_id, sample_id),
            )
            next_id += 1

    def _next_review_id(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COALESCE(MAX(review_id), 0) + 1 FROM recordings"
            ).fetchone()[0]
        )

    def setting(self, key: str) -> str:
        row = self.connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row["value"])

    def opaque_id(self, kind: str, raw_value: str, length: int = 16) -> str:
        salt = bytes.fromhex(self.setting("privacy_salt"))
        digest = hmac.new(salt, f"{kind}:{raw_value}".encode(), hashlib.sha256).hexdigest()[:length]
        return f"{kind}_{digest}"

    def begin_candidate(
        self,
        candidate: Candidate,
        collection_run_id: str,
        consent_status: str,
        language: str = "ru",
        consent_ref: str = "",
        _commit: bool = True,
    ) -> sqlite3.Row:
        existing = self.get_by_source(candidate)
        review_id = (
            int(existing["review_id"])
            if existing is not None and existing["review_id"] is not None
            else self._next_review_id()
        )
        identity = (
            f"{candidate.source_type}|{candidate.source_chat_id}|{candidate.source_message_id}"
        )
        source_recording_id = self.opaque_id("rec", identity)
        chat_ref = self.opaque_id("chat", f"{candidate.source_type}|{candidate.source_chat_id}", 12)
        message_ref = self.opaque_id("msg", identity, 12)
        sender_ref = (
            self.opaque_id("sender", str(candidate.sender_id), 12) if candidate.sender_id else ""
        )
        flags = ["speaker_unverified"]
        effective_consent = consent_status
        if candidate.forwarded:
            flags.append("forwarded_message")
            flags.append("consent_needs_review")
            effective_consent = "unknown"
            consent_ref = ""
        message_date = (
            candidate.message_date_utc.astimezone(UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        values = {
            "sample_id": source_recording_id,
            "review_id": review_id,
            "schema_version": SCHEMA_VERSION,
            "collection_run_id": collection_run_id,
            "source_recording_id": source_recording_id,
            "source_type": candidate.source_type,
            "source_chat_id": candidate.source_chat_id,
            "source_message_id": candidate.source_message_id,
            "chat_ref": chat_ref,
            "message_ref": message_ref,
            "message_date_utc": message_date,
            "media_type": candidate.media_type,
            "sender_ref": sender_ref,
            "forwarded": int(candidate.forwarded),
            "telegram_duration_ms": candidate.telegram_duration_ms,
            "task_type": "speaker_verification",
            "language": language,
            "consent_status": effective_consent,
            "consent_ref": consent_ref,
            "review_flags": "|".join(flags),
            "recording_group_id": source_recording_id,
            "split_group_id": source_recording_id,
            "dedupe_family_id": source_recording_id,
        }
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        self.connection.execute(
            f"INSERT OR IGNORE INTO recordings ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO collection_runs(run_id) VALUES (?)",
            (collection_run_id,),
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO collection_run_items(run_id, sample_id, position)
            SELECT ?, sample_id,
                   COALESCE((SELECT MAX(position) + 1
                               FROM collection_run_items WHERE run_id = ?), 1)
              FROM recordings
             WHERE source_type = ? AND source_chat_id = ? AND source_message_id = ?
            """,
            (
                collection_run_id,
                collection_run_id,
                candidate.source_type,
                candidate.source_chat_id,
                candidate.source_message_id,
            ),
        )
        if _commit:
            self.connection.commit()
        row = self.get_by_source(candidate)
        assert row is not None
        return row

    def begin_candidates(
        self,
        candidates: Iterable[Candidate],
        collection_run_id: str,
        consent_status: str,
        language: str = "ru",
        consent_ref: str = "",
    ) -> list[sqlite3.Row]:
        rows: list[sqlite3.Row] = []
        with self.connection:
            for candidate in candidates:
                rows.append(
                    self.begin_candidate(
                        candidate,
                        collection_run_id,
                        consent_status,
                        language,
                        consent_ref,
                        _commit=False,
                    )
                )
        return rows

    def get_by_source(self, candidate: Candidate) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM recordings
             WHERE source_type = ? AND source_chat_id = ? AND source_message_id = ?
            """,
            (
                candidate.source_type,
                candidate.source_chat_id,
                candidate.source_message_id,
            ),
        ).fetchone()

    def get(self, sample_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM recordings WHERE sample_id = ?", (sample_id,)
        ).fetchone()

    def get_by_review_id(self, review_id: int) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM recordings WHERE review_id = ?", (review_id,)
        ).fetchone()

    def update(self, sample_id: str, **values: Any) -> None:
        if not values:
            return
        allowed = {row[1] for row in self.connection.execute("PRAGMA table_info(recordings)")}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown state columns: {sorted(unknown)}")
        assignments = ", ".join(f"{column} = ?" for column in values)
        parameters = [*values.values(), sample_id]
        self.connection.execute(
            f"UPDATE recordings SET {assignments} WHERE sample_id = ?", parameters
        )
        self.connection.commit()

    def find_pcm_duplicate(self, pcm_sha256: str, excluding_sample_id: str) -> sqlite3.Row | None:
        if not pcm_sha256:
            return None
        return self.connection.execute(
            """
            SELECT * FROM recordings
             WHERE audio_pcm_sha256 = ? AND sample_id != ?
               AND processing_status IN ('indexed', 'excluded')
               AND dedupe_status = 'unique'
             ORDER BY processed_at_utc, sample_id
             LIMIT 1
            """,
            (pcm_sha256, excluding_sample_id),
        ).fetchone()

    def rows(self, collection_run_id: str | None = None) -> list[sqlite3.Row]:
        if collection_run_id:
            cursor = self.connection.execute(
                """
                SELECT recordings.*
                  FROM collection_run_items
                  JOIN recordings USING (sample_id)
                 WHERE collection_run_items.run_id = ?
                 ORDER BY collection_run_items.position, recordings.sample_id
                """,
                (collection_run_id,),
            )
        else:
            cursor = self.connection.execute(
                "SELECT * FROM recordings ORDER BY message_date_utc, sample_id"
            )
        return list(cursor.fetchall())

    def latest_run_id(self) -> str | None:
        row = self.connection.execute(
            """
            SELECT run_id
              FROM collection_runs
             ORDER BY rowid DESC
             LIMIT 1
            """
        ).fetchone()
        return str(row[0]) if row else None

    def is_first_run(self, run_id: str) -> bool:
        row = self.connection.execute(
            "SELECT run_id FROM collection_runs ORDER BY rowid ASC LIMIT 1"
        ).fetchone()
        return bool(row and str(row[0]) == run_id)

    def run_status(self, run_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT status FROM collection_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return str(row[0]) if row else None

    def mark_run_complete(self, run_id: str) -> None:
        self.connection.execute(
            "UPDATE collection_runs SET status = 'complete' WHERE run_id = ?",
            (run_id,),
        )
        self.connection.commit()

    def update_many(self, updates: Iterable[tuple[str, dict[str, Any]]]) -> None:
        allowed = {row[1] for row in self.connection.execute("PRAGMA table_info(recordings)")}
        with self.connection:
            for sample_id, values in updates:
                if not values:
                    continue
                unknown = set(values) - allowed
                if unknown:
                    raise ValueError(f"Unknown state columns: {sorted(unknown)}")
                assignments = ", ".join(f"{column} = ?" for column in values)
                parameters = [*values.values(), sample_id]
                self.connection.execute(
                    f"UPDATE recordings SET {assignments} WHERE sample_id = ?",
                    parameters,
                )
