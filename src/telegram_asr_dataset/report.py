from __future__ import annotations

import json
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .state import StateStore


def _round_seconds(milliseconds: int | float) -> float:
    return round(float(milliseconds) / 1000, 3)


def build_report(store: StateStore, collection_run_id: str) -> dict[str, Any]:
    rows = store.rows(collection_run_id)
    active_rows = [row for row in rows if row["processing_status"] != "purged"]
    durations = [int(row["audio_duration_ms"]) for row in active_rows if row["audio_duration_ms"]]
    media = Counter(str(row["media_type"]) for row in active_rows)
    chat_counts = Counter(str(row["chat_ref"]) for row in active_rows)
    qc_flags: Counter[str] = Counter()
    quality_decisions = Counter(str(row["quality_decision"]) for row in active_rows)
    quality_by_media: dict[str, Counter[str]] = {}
    quality_reasons: Counter[str] = Counter()
    quality_by_chat: dict[str, Counter[str]] = {}
    for row in active_rows:
        qc_flags.update(flag for flag in str(row["qc_flags"]).split("|") if flag)
        quality_reasons.update(flag for flag in str(row["quality_reasons"]).split("|") if flag)
        quality_by_chat.setdefault(str(row["chat_ref"]), Counter()).update(
            [str(row["quality_decision"])]
        )
        quality_by_media.setdefault(str(row["media_type"]), Counter()).update(
            [str(row["quality_decision"])]
        )

    decoded = sum(bool(row["normalized_audio_path"]) for row in active_rows)
    errors = sum(row["processing_status"] == "error" for row in active_rows)
    duplicates = sum(row["dedupe_status"] == "exact_duplicate" for row in active_rows)
    report: dict[str, Any] = {
        "collection_run_id": collection_run_id,
        "selected_messages": len(rows),
        "active_recordings": len(active_rows),
        "purged_recordings": len(rows) - len(active_rows),
        "decoded_audio": decoded,
        "errors": errors,
        "exact_duplicates": duplicates,
        "unique_recordings": sum(row["dedupe_status"] == "unique" for row in active_rows),
        "pending_review": sum(row["dataset_status"] == "pending_review" for row in active_rows),
        "ready_for_training": sum(row["dataset_status"] == "ready" for row in active_rows),
        "excluded": sum(row["dataset_status"] == "excluded" for row in rows),
        "excluded_video_notes": sum(
            row["exclusion_reason"] == "video_notes_disabled" for row in rows
        ),
        "speaker_unreviewed": sum(row["speaker_status"] == "unreviewed" for row in active_rows),
        "quality_decisions": dict(sorted(quality_decisions.items())),
        "quality_reasons": dict(sorted(quality_reasons.items())),
        "quality_by_chat": {
            chat: dict(sorted(counts.items())) for chat, counts in sorted(quality_by_chat.items())
        },
        "quality_by_media": {
            media_type: dict(sorted(counts.items()))
            for media_type, counts in sorted(quality_by_media.items())
        },
        "media_types": dict(sorted(media.items())),
        "chats": dict(sorted(chat_counts.items())),
        "qc_flags": dict(sorted(qc_flags.items())),
        "raw_bytes": sum(int(row["raw_size_bytes"] or 0) for row in active_rows),
        "extracted_bytes": sum(int(row["extracted_size_bytes"] or 0) for row in active_rows),
        "normalized_bytes": sum(int(row["normalized_size_bytes"] or 0) for row in active_rows),
        "total_duration_seconds": _round_seconds(sum(durations)),
        "external_uploads": 0,
    }
    if durations:
        report["duration_seconds"] = {
            "minimum": _round_seconds(min(durations)),
            "median": _round_seconds(statistics.median(durations)),
            "mean": _round_seconds(statistics.mean(durations)),
            "maximum": _round_seconds(max(durations)),
        }
    else:
        report["duration_seconds"] = {
            "minimum": 0,
            "median": 0,
            "mean": 0,
            "maximum": 0,
        }
    return report


def report_markdown(report: dict[str, Any], pilot: bool = False) -> str:
    durations = report["duration_seconds"]
    lines = [
        (
            "# Отчёт пилотного датасета speaker verification"
            if pilot
            else "# Отчёт датасета speaker verification"
        ),
        "",
        f"Идентификатор запуска: `{report['collection_run_id']}`",
        "",
        "## Объём",
        "",
        f"- Выбрано подходящих сообщений: {report['selected_messages']}",
        f"- Осталось локальных записей: {report['active_recordings']}",
        f"- Удалено по ручной проверке/согласию: {report['purged_recordings']}",
        f"- Успешно декодировано: {report['decoded_audio']}",
        f"- Уникальных записей: {report['unique_recordings']}",
        f"- Точных дубликатов: {report['exact_duplicates']}",
        f"- Ошибок: {report['errors']}",
        f"- Общая длительность: {report['total_duration_seconds']} с",
        f"- Размер исходников: {report['raw_bytes']} байт",
        f"- Размер извлечённого аудио из кружков: {report['extracted_bytes']} байт",
        f"- Размер нормализованного WAV: {report['normalized_bytes']} байт",
        "",
        "## Длительность записи",
        "",
        f"- Минимум: {durations['minimum']} с",
        f"- Медиана: {durations['median']} с",
        f"- Среднее: {durations['mean']} с",
        f"- Максимум: {durations['maximum']} с",
        "",
        "## Разметка и проверка",
        "",
        f"- Автоматическое качество: {json.dumps(report['quality_decisions'], ensure_ascii=False)}",
        f"- Качество по типу медиа: {json.dumps(report['quality_by_media'], ensure_ascii=False)}",
        f"- Ожидают ручной проверки: {report['pending_review']}",
        f"- Исключены из обучающей выборки: {report['excluded']}",
        f"- Из них кружков по текущей политике: {report['excluded_video_notes']}",
        f"- Говорящий не подтверждён вручную: {report['speaker_unreviewed']}",
        f"- Готовы к обучению: {report['ready_for_training']}",
        "",
        "Отправитель сообщения не считается автоматически говорящим. Строка "
        "становится готовой только после ручного подтверждения одного говорящего, "
        "его личности и сессии записи.",
        "",
        "## Состав",
        "",
        f"- Типы медиа: {json.dumps(report['media_types'], ensure_ascii=False)}",
        f"- Псевдонимы чатов: {json.dumps(report['chats'], ensure_ascii=False)}",
        f"- Флаги качества: {json.dumps(report['qc_flags'], ensure_ascii=False)}",
        f"- Причины решений качества: {json.dumps(report['quality_reasons'], ensure_ascii=False)}",
        f"- Качество по чатам: {json.dumps(report['quality_by_chat'], ensure_ascii=False)}",
        "",
        "## Приватность",
        "",
        "- Загрузок записей во внешние сервисы: 0",
        "- Обработка и контроль качества выполнены локально.",
        "",
    ]
    return "\n".join(lines)


def save_report(
    store: StateStore,
    reports_directory: Path,
    collection_run_id: str,
    pilot: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    reports_directory.mkdir(parents=True, exist_ok=True)
    report = build_report(store, collection_run_id)
    prefix = "pilot_20" if pilot else "collection_report"
    markdown_path = reports_directory / f"{prefix}_{collection_run_id}.md"
    json_path = reports_directory / f"{prefix}_{collection_run_id}.json"
    for path, content in (
        (markdown_path, report_markdown(report, pilot=pilot)),
        (json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
    ):
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=path.name + ".",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return markdown_path, json_path, report
