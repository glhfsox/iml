from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Candidate:
    """A single Telegram media message before local processing.

    Raw Telegram identifiers stay in the private SQLite state.  Public metadata
    receives deterministic pseudonyms derived from these values.
    """

    source_type: str
    source_chat_id: str
    source_message_id: str
    message_date_utc: datetime
    media_type: str
    sender_id: str | None
    extension: str
    forwarded: bool = False
    telegram_duration_ms: int | None = None

    def __post_init__(self) -> None:
        if self.source_type not in {"telegram_api", "telegram_export", "local_directory"}:
            raise ValueError(f"Unsupported source_type: {self.source_type}")
        if self.media_type not in {"voice", "video_note"}:
            raise ValueError(f"Unsupported media_type: {self.media_type}")
        if self.message_date_utc.tzinfo is None:
            raise ValueError("message_date_utc must be timezone-aware")


@dataclass(frozen=True, slots=True)
class AudioMetrics:
    duration_ms: int
    sample_rate_hz: int
    channels: int
    sample_width_bits: int
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    active_audio_ms: int
    active_audio_ratio: float
    estimated_snr_db: float | None
    pcm_sha256: str
    qc_status: str
    qc_flags: tuple[str, ...]
    quality_score: int
    quality_decision: str
    quality_reasons: tuple[str, ...]
