from __future__ import annotations

import getpass
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Candidate


class TelegramDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TelegramDialog:
    index: int
    title: str
    source_chat_id: str
    entity: Any
    is_personal: bool


@dataclass(frozen=True, slots=True)
class TelegramItem:
    candidate: Candidate
    message: Any


def _telethon():
    try:
        from telethon import TelegramClient, errors
    except ImportError as exc:  # pragma: no cover - environment specific
        raise TelegramDependencyError(
            "Telethon is not installed. Install the project dependencies first."
        ) from exc
    return TelegramClient, errors


class TelegramSource:
    def __init__(self, session_path: Path):
        TelegramClient, errors = _telethon()
        self._errors = errors
        api_id_text = os.environ.get("TELEGRAM_API_ID", "").strip()
        api_hash = os.environ.get("TELEGRAM_API_HASH", "").strip()
        if not api_id_text:
            api_id_text = getpass.getpass("Telegram API ID (ввод скрыт): ").strip()
        if not api_hash:
            api_hash = getpass.getpass("Telegram API hash (ввод скрыт): ").strip()
        try:
            api_id = int(api_id_text)
        except ValueError as exc:
            raise ValueError("Telegram API ID must be an integer") from exc
        if not api_hash:
            raise ValueError("Telegram API hash is required")

        logging.getLogger("telethon").setLevel(logging.CRITICAL)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path = session_path
        self.client = TelegramClient(str(session_path), api_id, api_hash)

    async def connect_and_authorize(self) -> None:
        await self.client.connect()
        if await self.client.is_user_authorized():
            return
        phone = getpass.getpass("Номер телефона Telegram (ввод скрыт): ").strip()
        if not phone:
            raise ValueError("A Telegram phone number is required for first login")
        await self.client.send_code_request(phone)
        login_code = getpass.getpass("Код входа Telegram (ввод скрыт): ").strip()
        try:
            await self.client.sign_in(phone=phone, code=login_code)
        except self._errors.SessionPasswordNeededError:
            password = getpass.getpass("Пароль двухфакторной защиты (ввод скрыт): ")
            await self.client.sign_in(password=password)

    async def disconnect(self) -> None:
        await self.client.disconnect()
        session_file = Path(str(self.session_path) + ".session")
        if self.session_path.suffix == ".session":
            session_file = self.session_path
        if session_file.exists():
            try:
                os.chmod(session_file, 0o600)
            except OSError:
                pass

    async def dialogs(self) -> list[TelegramDialog]:
        result: list[TelegramDialog] = []
        async for dialog in self.client.iter_dialogs():
            result.append(
                TelegramDialog(
                    index=len(result) + 1,
                    title=str(dialog.name or "Без названия"),
                    source_chat_id=str(dialog.id),
                    entity=dialog.entity,
                    is_personal=bool(getattr(dialog, "is_user", False)),
                )
            )
        return result

    async def voice_counts(
        self,
        dialogs: list[TelegramDialog],
        start_utc: datetime,
        end_exclusive_utc: datetime,
        minimum_duration_seconds: float = 2.0,
        maximum_duration_seconds: float = 45.0,
    ) -> list[tuple[TelegramDialog, int, int, int, int]]:
        """Count voice messages without downloading media."""

        result: list[tuple[TelegramDialog, int, int, int, int]] = []
        for dialog in dialogs:
            eligible = outgoing = incoming = all_voice = 0
            async for message in self.client.iter_messages(
                dialog.entity, offset_date=end_exclusive_utc
            ):
                message_date = message.date
                if message_date.tzinfo is None:
                    message_date = message_date.replace(tzinfo=UTC)
                message_date = message_date.astimezone(UTC)
                if message_date >= end_exclusive_utc:
                    continue
                if message_date < start_utc:
                    break
                if self._media_type(message) != "voice":
                    continue
                all_voice += 1
                duration = getattr(getattr(message, "file", None), "duration", None)
                try:
                    duration_seconds = float(duration)
                except (TypeError, ValueError):
                    duration_seconds = None
                if duration_seconds is not None and not (
                    minimum_duration_seconds <= duration_seconds <= maximum_duration_seconds
                ):
                    continue
                eligible += 1
                if bool(getattr(message, "out", False)):
                    outgoing += 1
                else:
                    incoming += 1
            if all_voice:
                result.append((dialog, eligible, outgoing, incoming, all_voice))
        return result

    @staticmethod
    def _media_type(message: Any) -> str | None:
        if getattr(message, "voice", None) is not None:
            return "voice"
        if getattr(message, "video_note", None) is not None:
            return "video_note"
        return None

    async def items(
        self,
        dialogs: list[TelegramDialog],
        start_utc: datetime,
        end_exclusive_utc: datetime,
        include_video_notes: bool = False,
    ) -> list[TelegramItem]:
        result: list[TelegramItem] = []
        for dialog in dialogs:
            async for message in self.client.iter_messages(
                dialog.entity, offset_date=end_exclusive_utc
            ):
                message_date = message.date
                if message_date.tzinfo is None:
                    message_date = message_date.replace(tzinfo=UTC)
                message_date = message_date.astimezone(UTC)
                if message_date >= end_exclusive_utc:
                    continue
                if message_date < start_utc:
                    break
                media_type = self._media_type(message)
                if media_type is None:
                    continue
                if media_type == "video_note" and not include_video_notes:
                    continue
                file_info = getattr(message, "file", None)
                extension = str(getattr(file_info, "ext", "") or "")
                duration = getattr(file_info, "duration", None)
                try:
                    duration_ms = round(float(duration) * 1000) if duration else None
                except (TypeError, ValueError):
                    duration_ms = None
                candidate = Candidate(
                    source_type="telegram_api",
                    source_chat_id=dialog.source_chat_id,
                    source_message_id=str(message.id),
                    message_date_utc=message_date,
                    media_type=media_type,
                    sender_id=(str(message.sender_id) if message.sender_id else None),
                    extension=extension,
                    forwarded=getattr(message, "fwd_from", None) is not None,
                    telegram_duration_ms=duration_ms,
                )
                result.append(TelegramItem(candidate, message))
        return result

    async def download(self, item: TelegramItem, temporary_path: Path) -> Path:
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.unlink(missing_ok=True)
        downloaded = await self.client.download_media(item.message, file=str(temporary_path))
        if not downloaded:
            raise RuntimeError("Telegram returned no downloaded file")
        downloaded_path = Path(downloaded).resolve()
        if not downloaded_path.is_file() or downloaded_path.stat().st_size == 0:
            raise RuntimeError("Telegram download is missing or empty")
        return downloaded_path
