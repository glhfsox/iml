from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from telegram_asr_dataset.telegram_source import TelegramDialog, TelegramSource


class FakeClient:
    def __init__(self, messages):
        self.messages = messages

    async def iter_messages(self, _entity, offset_date):
        del offset_date
        for message in self.messages:
            yield message


def message(message_id: int, *, outgoing: bool):
    return SimpleNamespace(
        id=message_id,
        date=datetime(2025, 6, message_id, tzinfo=UTC),
        voice=object(),
        video_note=None,
        out=outgoing,
        file=SimpleNamespace(ext=".ogg", duration=5),
        sender_id=message_id,
        fwd_from=None,
    )


def source_with(messages) -> TelegramSource:
    source = object.__new__(TelegramSource)
    source.client = FakeClient(messages)
    return source


def test_items_can_select_only_incoming_messages() -> None:
    source = source_with([message(3, outgoing=True), message(2, outgoing=False)])
    dialog = TelegramDialog(1, "Test", "chat", object(), True)
    items = asyncio.run(
        source.items(
            [dialog],
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            direction="incoming",
        )
    )
    assert [item.candidate.source_message_id for item in items] == ["2"]


def test_items_reject_invalid_direction() -> None:
    source = source_with([])
    with pytest.raises(ValueError, match="Unsupported message direction"):
        asyncio.run(
            source.items(
                [],
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=1),
                direction="sideways",
            )
        )
