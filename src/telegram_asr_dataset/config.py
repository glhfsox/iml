from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path
    data: Path
    private: Path
    raw_voice: Path
    raw_video: Path
    extracted: Path
    normalized: Path
    reports: Path
    labels: Path
    state_db: Path
    metadata_csv: Path
    downloads: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        root = root.resolve()
        data = root / "data"
        return cls(
            root=root,
            data=data,
            private=data / "private",
            raw_voice=data / "raw" / "voice",
            raw_video=data / "raw" / "video_notes",
            extracted=data / "extracted",
            normalized=data / "normalized",
            reports=data / "reports",
            labels=data / "labels",
            state_db=data / "state" / "collector.sqlite3",
            metadata_csv=data / "metadata.csv",
            downloads=root / "work" / "downloads",
        )

    def create(self) -> None:
        for directory in (
            self.data,
            self.private,
            self.data / "raw",
            self.raw_voice,
            self.raw_video,
            self.extracted,
            self.normalized,
            self.reports,
            self.labels,
            self.state_db.parent,
            self.downloads,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass


def parse_date_range(
    start_text: str, end_text: str, timezone_name: str
) -> tuple[datetime, datetime]:
    """Return a UTC half-open interval for two inclusive local calendar dates."""

    try:
        start_day = date.fromisoformat(start_text)
        end_day = date.fromisoformat(end_text)
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format") from exc
    if end_day < start_day:
        raise ValueError("end date must not be earlier than start date")
    try:
        local_tz = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc

    start_local = datetime.combine(start_day, time.min, tzinfo=local_tz)
    end_exclusive_local = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=local_tz)
    return (
        start_local.astimezone(UTC),
        end_exclusive_local.astimezone(UTC),
    )


def safe_extension(extension: str, fallback: str = ".bin") -> str:
    extension = extension.lower().strip()
    if extension and not extension.startswith("."):
        extension = "." + extension
    body = extension[1:]
    if not body or len(body) > 10 or not body.isalnum():
        return fallback
    return extension
