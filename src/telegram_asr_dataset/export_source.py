from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .models import Candidate


@dataclass(frozen=True, slots=True)
class ExportChat:
    source_chat_id: str
    title: str
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ExportItem:
    candidate: Candidate
    media_path: Path | None
    path_error: str = ""


def load_export_chats(export_directory: Path) -> list[ExportChat]:
    export_directory = export_directory.expanduser().resolve()
    result_path = export_directory / "result.json"
    if not result_path.is_file():
        raise ValueError(f"Telegram export JSON was not found: {result_path}")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Telegram export JSON: {exc}") from exc

    raw_chats: list[dict[str, Any]]
    if isinstance(payload.get("chats"), dict):
        raw_chats = payload["chats"].get("list", [])
    elif isinstance(payload.get("messages"), list):
        raw_chats = [payload]
    else:
        raise ValueError("Unsupported Telegram export structure")

    chats: list[ExportChat] = []
    for position, raw_chat in enumerate(raw_chats, start=1):
        messages = raw_chat.get("messages", [])
        if not isinstance(messages, list):
            continue
        raw_id = raw_chat.get("id")
        title = str(raw_chat.get("name") or f"Chat {position}")
        source_chat_id = str(raw_id if raw_id is not None else f"name:{title}")
        chats.append(ExportChat(source_chat_id, title, tuple(messages)))
    return chats


def _message_datetime(message: dict[str, Any], export_timezone: str) -> datetime:
    unix_value = message.get("date_unixtime")
    if unix_value not in {None, ""}:
        try:
            return datetime.fromtimestamp(int(unix_value), tz=UTC)
        except (TypeError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid date_unixtime: {unix_value!r}") from exc

    date_value = str(message.get("date") or "")
    if not date_value:
        raise ValueError("Message has no timestamp")
    try:
        parsed = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid message date: {date_value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(export_timezone))
    return parsed.astimezone(UTC)


def _safe_export_path(export_directory: Path, value: str) -> Path:
    if not value or value.startswith("("):
        raise ValueError("Media was not included in the Telegram export")
    root = export_directory.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Export media path escapes the export directory") from exc
    if not candidate.is_file():
        raise ValueError("Export media file is missing")
    return candidate


def export_items(
    export_directory: Path,
    chats: list[ExportChat],
    start_utc: datetime,
    end_exclusive_utc: datetime,
    export_timezone: str = "Europe/Warsaw",
    include_video_notes: bool = False,
) -> list[ExportItem]:
    export_directory = export_directory.expanduser().resolve()
    items: list[ExportItem] = []
    for chat in chats:
        for message in chat.messages:
            media_value = str(message.get("media_type") or "")
            if media_value == "voice_message":
                media_type = "voice"
            elif media_value == "video_message":
                media_type = "video_note"
            else:
                continue
            if media_type == "video_note" and not include_video_notes:
                continue
            try:
                message_date = _message_datetime(message, export_timezone)
            except ValueError:
                continue
            if not (start_utc <= message_date < end_exclusive_utc):
                continue

            file_value = str(message.get("file") or "")
            media_path: Path | None
            path_error = ""
            try:
                media_path = _safe_export_path(export_directory, file_value)
            except ValueError as exc:
                media_path = None
                path_error = str(exc)
            extension = Path(file_value).suffix or (".ogg" if media_type == "voice" else ".mp4")
            duration = message.get("duration_seconds")
            try:
                duration_ms = round(float(duration) * 1000) if duration is not None else None
            except (TypeError, ValueError):
                duration_ms = None
            message_id = str(message.get("id") or "")
            if not message_id:
                continue
            candidate = Candidate(
                source_type="telegram_export",
                source_chat_id=chat.source_chat_id,
                source_message_id=message_id,
                message_date_utc=message_date,
                media_type=media_type,
                sender_id=(str(message.get("from_id")) if message.get("from_id") else None),
                extension=extension,
                forwarded=bool(message.get("forwarded_from")),
                telegram_duration_ms=duration_ms,
            )
            items.append(ExportItem(candidate, media_path, path_error))
    return items
