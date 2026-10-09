from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .audio import AudioProcessingError, find_ffmpeg
from .config import ProjectPaths, parse_date_range
from .export_source import ExportChat, export_items, load_export_chats
from .metadata import apply_labels_csv, write_labels_template, write_metadata_csv
from .pipeline import DatasetProject, new_collection_run_id
from .report import save_report
from .split import assert_group_safe, assign_splits
from .telegram_source import TelegramDialog, TelegramSource
from .trials import build_verification_trials


def _parse_indices(text: str, maximum: int) -> list[int]:
    selected: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("At least one chat must be selected")
    if min(selected) < 1 or max(selected) > maximum:
        raise ValueError(f"Chat selection must be between 1 and {maximum}")
    return sorted(selected)


def _selection_text(provided: list[str] | None) -> str:
    if provided:
        return ",".join(provided)
    return input("Выберите номера чатов (например 1,3-4): ").strip()


def _select_chats(
    choices: Sequence[ExportChat | TelegramDialog],
    project: DatasetProject,
    source_type: str,
    provided_indices: list[str] | None,
    provided_refs: list[str] | None,
    assume_yes: bool,
    expected_count: int | None = None,
) -> list[Any]:
    if not choices:
        raise ValueError("No chats were found")
    by_ref: dict[str, Any] = {}
    for index, chat in enumerate(choices, start=1):
        chat_ref = project.store.opaque_id("chat", f"{source_type}|{chat.source_chat_id}", 12)
        by_ref[chat_ref] = chat
        print(f"{index:>3}. {chat.title} [{chat_ref}]")
    if provided_refs:
        requested_refs = [
            value.strip() for group in provided_refs for value in group.split(",") if value.strip()
        ]
        unknown = [value for value in requested_refs if value not in by_ref]
        if unknown:
            raise ValueError(f"Unknown chat_ref values: {unknown}")
        selected = [by_ref[value] for value in requested_refs]
    else:
        indices = _parse_indices(_selection_text(provided_indices), len(choices))
        selected = [choices[index - 1] for index in indices]
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(
            f"Для проекта выберите ровно {expected_count} чатов; сейчас выбрано {len(selected)}"
        )
    print("Выбраны чаты:")
    for chat in selected:
        print(f"- {chat.title}")
    if not assume_yes:
        answer = input("Подтвердить этот список чатов? [y/N]: ").strip().casefold()
        if answer not in {"y", "yes", "д", "да"}:
            raise ValueError("Chat selection was not confirmed")
    return selected


def _dates(args: argparse.Namespace) -> tuple[Any, Any]:
    start = args.start or input("Начальная дата YYYY-MM-DD: ").strip()
    end = args.end or input("Конечная дата YYYY-MM-DD включительно: ").strip()
    return parse_date_range(start, end, args.timezone)


def _consent_details(args: argparse.Namespace) -> tuple[str, str]:
    if args.consent_status:
        status = str(args.consent_status)
    else:
        answer = (
            input("Согласие людей в выбранных записях подтверждено для учебного проекта? [y/N]: ")
            .strip()
            .casefold()
        )
        status = "confirmed_other" if answer in {"y", "yes", "д", "да"} else "unknown"
    reference = str(args.consent_ref or "").strip()
    if status == "confirmed_other" and not reference:
        reference = input(
            "Введите непрозрачный код записи о согласии (без имён; например consent_01): "
        ).strip()
    if status == "confirmed_self" and not reference:
        reference = "self_declaration"
    return status, reference


def _sort_items(items: list[Any], order: str) -> list[Any]:
    reverse = order == "newest"
    return sorted(
        items,
        key=lambda item: (
            item.candidate.message_date_utc,
            item.candidate.source_chat_id,
            item.candidate.source_message_id,
        ),
        reverse=reverse,
    )


def _ordered_items(items: list[Any], order: str, sampling: str) -> list[Any]:
    ordered = _sort_items(items, order)
    if sampling == "global":
        return ordered
    # Sender is only a sampling hint, never a verified speaker label. Bucketing
    # by chat and sender prevents a batch from being dominated by outgoing audio.
    by_chat_sender: dict[tuple[str, str], list[Any]] = {}
    for item in ordered:
        key = (
            item.candidate.source_chat_id,
            str(item.candidate.sender_id or "unknown_sender"),
        )
        by_chat_sender.setdefault(key, []).append(item)
    balanced: list[Any] = []
    buckets = sorted(by_chat_sender)
    position = 0
    while True:
        added = False
        for bucket in buckets:
            bucket_items = by_chat_sender[bucket]
            if position < len(bucket_items):
                balanced.append(bucket_items[position])
                added = True
        if not added:
            return balanced
        position += 1


def _items_needing_collection(
    project: DatasetProject, items: list[Any], retry_errors: bool = False
) -> list[Any]:
    final_statuses = {"indexed", "excluded", "purged"}
    if not retry_errors:
        final_statuses.add("error")
    pending: list[Any] = []
    for item in items:
        row = project.store.get_by_source(item.candidate)
        if row is None or str(row["processing_status"]) not in final_statuses:
            pending.append(item)
    return pending


def _filter_by_duration(
    items: list[Any], minimum_seconds: float, maximum_seconds: float
) -> tuple[list[Any], int]:
    if minimum_seconds < 0 or maximum_seconds <= minimum_seconds:
        raise ValueError("Invalid duration range")
    kept: list[Any] = []
    skipped = 0
    minimum_ms = round(minimum_seconds * 1000)
    maximum_ms = round(maximum_seconds * 1000)
    for item in items:
        duration_ms = item.candidate.telegram_duration_ms
        if duration_ms is not None and not (minimum_ms <= duration_ms <= maximum_ms):
            skipped += 1
            continue
        kept.append(item)
    return kept, skipped


def _ensure_selected_chat_coverage(selected_chats: list[Any], items: list[Any]) -> None:
    contributing = {item.candidate.source_chat_id for item in items}
    missing = [chat.title for chat in selected_chats if chat.source_chat_id not in contributing]
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(
            "В выбранных чатах нет подходящих голосовых длительностью в заданном "
            f"диапазоне: {formatted}. Выберите замену или измените пороги длительности."
        )


def _ensure_first_run_is_pilot(project: DatasetProject, limit: int, next_batch: bool) -> None:
    if project.store.latest_run_id() is None and (limit != 20 or next_batch):
        raise ValueError("The first collection run must use the 20-message pilot limit")


def _source_key_from_item(item: Any) -> tuple[str, str, str]:
    candidate = item.candidate
    return (
        candidate.source_type,
        candidate.source_chat_id,
        candidate.source_message_id,
    )


def _source_key_from_row(row: Any) -> tuple[str, str, str]:
    return (
        str(row["source_type"]),
        str(row["source_chat_id"]),
        str(row["source_message_id"]),
    )


def _select_collection_batch(
    project: DatasetProject,
    items: list[Any],
    limit: int,
    order: str,
    sampling: str,
    next_batch: bool,
    retry_errors: bool,
) -> tuple[str, list[Any], list[Any], bool]:
    """Resume the latest interrupted fixed batch, or create a new fixed batch."""

    if limit < 1:
        raise ValueError("limit must be positive")
    latest_run_id = project.store.latest_run_id()
    if latest_run_id and project.store.run_status(latest_run_id) == "processing":
        run_rows = project.store.rows(latest_run_id)
        if not run_rows:
            raise ValueError("The latest collection run has no registered items")
        pending_rows = [
            row
            for row in run_rows
            if str(row["processing_status"])
            not in ({"indexed", "excluded", "purged"} | ({"error"} if not retry_errors else set()))
        ]
        item_by_source = {_source_key_from_item(item): item for item in items}
        try:
            window = [item_by_source[_source_key_from_row(row)] for row in run_rows]
        except KeyError as exc:
            raise ValueError(
                "The latest collection run is incomplete, but its original chat/date "
                "selection is unavailable. Re-run with the same chats and period."
            ) from exc
        pending_keys = {_source_key_from_row(row) for row in pending_rows}
        pending = [item for item in window if _source_key_from_item(item) in pending_keys]
        print(f"Продолжаю незавершённый запуск {latest_run_id}.")
        return latest_run_id, window, pending, True

    ordered = _ordered_items(items, order, sampling)
    if next_batch:
        window = _items_needing_collection(project, ordered, retry_errors=retry_errors)[:limit]
    else:
        window = ordered[:limit]
    pending = _items_needing_collection(project, window, retry_errors=retry_errors)
    return new_collection_run_id(), window, pending, False


def _finish_collection(project: DatasetProject, run_id: str, selected_count: int) -> None:
    project.export_metadata()
    labels_path = project.paths.labels / f"speaker_labels_{run_id}.csv"
    write_labels_template(
        project.store,
        labels_path,
        run_id,
        speaker_aliases_path=project.paths.private / "speaker_aliases.csv",
    )
    is_pilot = project.store.is_first_run(run_id)
    markdown_path, _, report = save_report(
        project.store, project.paths.reports, run_id, pilot=is_pilot
    )
    project.store.mark_run_complete(run_id)
    print()
    print(f"Выбрано сообщений: {selected_count}")
    print(f"Декодировано: {report['decoded_audio']}")
    print(f"Ошибок: {report['errors']}; дубликатов: {report['exact_duplicates']}")
    print(f"Общая длительность: {report['total_duration_seconds']} с")
    print(f"Metadata: {project.paths.metadata_csv}")
    print(f"Таблица проверки говорящих: {labels_path}")
    print(f"Отчёт: {markdown_path}")
    if is_pilot and selected_count < 20:
        print("В выбранном интервале найдено меньше 20 подходящих сообщений.")


def command_doctor(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_root(Path(args.project_root))
    try:
        ffmpeg = find_ffmpeg()
        telethon_version = importlib.metadata.version("Telethon")
        print("Локальная среда готова.")
        print(f"FFmpeg: {ffmpeg}")
        print(f"Telethon: {telethon_version}")
        print(f"Приватные данные: {paths.data}")
        print("Внешняя загрузка аудио: не используется")
        return 0
    except (AudioProcessingError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"Среда ещё не готова: {exc}", file=sys.stderr)
        return 1


def command_collect_export(args: argparse.Namespace) -> int:
    start_utc, end_utc = _dates(args)
    export_directory = Path(args.export_dir).expanduser().resolve()
    with DatasetProject(Path(args.project_root)) as project:
        all_chats = load_export_chats(export_directory)
        selected_chats = _select_chats(
            all_chats,
            project,
            "telegram_export",
            args.chat_index,
            args.chat_ref,
            args.yes,
            args.expected_chat_count,
        )
        consent_status, consent_ref = _consent_details(args)
        items = export_items(
            export_directory,
            selected_chats,
            start_utc,
            end_utc,
            export_timezone=args.timezone,
            include_video_notes=args.include_video_notes,
        )
        items, skipped_by_duration = _filter_by_duration(
            items, args.min_duration, args.max_duration
        )
        if skipped_by_duration:
            print(f"Пропущено по длительности до скачивания: {skipped_by_duration}")
        _ensure_selected_chat_coverage(selected_chats, items)
        if not items:
            raise ValueError("No voice messages matched the filters")
        _ensure_first_run_is_pilot(project, args.limit, args.next_batch)
        run_id, selected_items, pending_items, resuming = _select_collection_batch(
            project,
            items,
            args.limit,
            args.order,
            args.sampling,
            args.next_batch,
            args.retry_errors,
        )
        if not pending_items:
            if resuming:
                _finish_collection(project, run_id, len(selected_items))
                return 0
            print("Все подходящие сообщения уже обработаны или были отозваны.")
            return 0
        # Fix run membership before processing so Ctrl-C can resume the same batch.
        project.begin_batch(
            [item.candidate for item in selected_items],
            run_id,
            consent_status,
            args.language,
            consent_ref,
        )
        for position, item in enumerate(pending_items, start=1):
            print(f"[{position}/{len(pending_items)}] {item.candidate.media_type}")
            if item.media_path is None:
                row = project.store.get_by_source(item.candidate)
                assert row is not None
                project.mark_error(str(row["sample_id"]), "export_media_missing", item.path_error)
                continue
            project.import_local_file(
                item.candidate,
                item.media_path,
                run_id,
                consent_status,
                args.language,
                consent_ref,
            )
        _finish_collection(project, run_id, len(selected_items))
    return 0


async def _collect_telegram_async(args: argparse.Namespace) -> int:
    start_utc, end_utc = _dates(args)
    project = DatasetProject(Path(args.project_root))
    _ensure_first_run_is_pilot(project, args.limit, args.next_batch)
    source = TelegramSource(project.paths.private / "telegram")
    try:
        await source.connect_and_authorize()
        dialogs = await source.dialogs()
        selected_dialogs = _select_chats(
            dialogs,
            project,
            "telegram_api",
            args.chat_index,
            args.chat_ref,
            args.yes,
            args.expected_chat_count,
        )
        consent_status, consent_ref = _consent_details(args)
        media_description = (
            "голосовые сообщения и видеокружки"
            if args.include_video_notes
            else "только голосовые сообщения"
        )
        print(f"Ищу {media_description} в выбранном периоде…")
        items = await source.items(
            selected_dialogs,
            start_utc,
            end_utc,
            include_video_notes=args.include_video_notes,
        )
        items, skipped_by_duration = _filter_by_duration(
            items, args.min_duration, args.max_duration
        )
        if skipped_by_duration:
            print(f"Пропущено по длительности до скачивания: {skipped_by_duration}")
        _ensure_selected_chat_coverage(selected_dialogs, items)
        if not items:
            raise ValueError("No voice messages matched the filters")
        run_id, selected_items, pending_items, resuming = _select_collection_batch(
            project,
            items,
            args.limit,
            args.order,
            args.sampling,
            args.next_batch,
            args.retry_errors,
        )
        if not pending_items:
            if resuming:
                _finish_collection(project, run_id, len(selected_items))
                return 0
            print("Все подходящие сообщения уже обработаны или были отозваны.")
            return 0

        project.begin_batch(
            [item.candidate for item in selected_items],
            run_id,
            consent_status,
            args.language,
            consent_ref,
        )
        for position, item in enumerate(pending_items, start=1):
            print(f"[{position}/{len(pending_items)}] {item.candidate.media_type}")
            row = project.store.get_by_source(item.candidate)
            assert row is not None
            sample_id = str(row["sample_id"])
            raw_path = str(row["raw_path"])
            if raw_path and project.absolute(raw_path).is_file():
                project.process_raw(sample_id)
                continue
            temporary = project.temporary_download_path(item.candidate, sample_id)
            try:
                downloaded = await source.download(item, temporary)
                project.import_local_file(
                    item.candidate,
                    downloaded,
                    run_id,
                    consent_status,
                    args.language,
                    consent_ref,
                    move=True,
                )
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                project.mark_error(sample_id, "telegram_download_failed", str(exc))
        _finish_collection(project, run_id, len(selected_items))
        return 0
    finally:
        await source.disconnect()
        project.close()


def command_collect_telegram(args: argparse.Namespace) -> int:
    return asyncio.run(_collect_telegram_async(args))


async def _rank_chats_async(args: argparse.Namespace) -> int:
    start_utc, end_utc = _dates(args)
    if args.min_duration < 0 or args.max_duration <= args.min_duration:
        raise ValueError("Invalid duration range")
    if args.top < 1:
        raise ValueError("top must be positive")
    project = DatasetProject(Path(args.project_root))
    source = TelegramSource(project.paths.private / "telegram")
    try:
        await source.connect_and_authorize()
        dialogs = [dialog for dialog in await source.dialogs() if dialog.is_personal]
        print("Считаю голосовые в личных чатах; медиа не скачиваются…")
        counts = await source.voice_counts(
            dialogs,
            start_utc,
            end_utc,
            args.min_duration,
            args.max_duration,
        )
        counts.sort(key=lambda item: (-item[1], item[0].title.casefold()))
        print("\nЛичные чаты с наибольшим числом голосовых:")
        for position, (dialog, eligible, outgoing, incoming, all_voice) in enumerate(
            counts[: args.top], start=1
        ):
            chat_ref = project.store.opaque_id("chat", f"telegram_api|{dialog.source_chat_id}", 12)
            print(
                f"{position:>2}. {dialog.title} [{chat_ref}] — подходящих "
                f"{eligible} из {all_voice}, ваших {outgoing}, "
                f"собеседника {incoming}"
            )
        print("\nВыберите только чаты участников, давших согласие; желательно с обеими сторонами.")
        return 0
    finally:
        await source.disconnect()
        project.close()


def command_rank_chats(args: argparse.Namespace) -> int:
    return asyncio.run(_rank_chats_async(args))


def command_report(args: argparse.Namespace) -> int:
    with DatasetProject(Path(args.project_root)) as project:
        run_id = args.run_id or project.store.latest_run_id()
        if not run_id:
            raise ValueError("There is no collection run to report")
        markdown_path, json_path, report = save_report(
            project.store,
            project.paths.reports,
            run_id,
            pilot=project.store.is_first_run(run_id),
        )
        print(f"Сообщений: {report['selected_messages']}")
        print(f"Длительность: {report['total_duration_seconds']} с")
        print(f"Отчёты: {markdown_path} and {json_path}")
    return 0


def command_export_labels(args: argparse.Namespace) -> int:
    with DatasetProject(Path(args.project_root)) as project:
        run_id = args.run_id or project.store.latest_run_id()
        destination = project.paths.labels / f"speaker_labels_{run_id or 'all'}.csv"
        write_labels_template(
            project.store,
            destination,
            run_id,
            speaker_aliases_path=project.paths.private / "speaker_aliases.csv",
        )
        print(f"Таблица проверки говорящих: {destination}")
    return 0


def command_apply_labels(args: argparse.Namespace) -> int:
    with DatasetProject(Path(args.project_root)) as project:
        changed = apply_labels_csv(project.store, Path(args.labels).expanduser().resolve())
        write_metadata_csv(project.store, project.paths.metadata_csv)
        print(f"Обновлено строк: {changed}")
        print(f"Metadata: {project.paths.metadata_csv}")
    return 0


def command_split(args: argparse.Namespace) -> int:
    with DatasetProject(Path(args.project_root)) as project:
        counts = assign_splits(
            project.store,
            seed=args.seed,
            train_ratio=args.train,
            validation_ratio=args.validation,
            test_ratio=args.test,
        )
        assert_group_safe(project.store)
        project.export_metadata()
        print(
            "Разбиение готово: "
            f"train={counts['train']}, validation={counts['validation']}, test={counts['test']}"
        )
    return 0


def command_build_trials(args: argparse.Namespace) -> int:
    with DatasetProject(Path(args.project_root)) as project:
        destination = project.paths.data / "trials.csv"
        counts = build_verification_trials(project.store, destination, seed=args.seed)
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"Verification trials готовы: {details}")
        print(f"Таблица: {destination}")
    return 0


def command_purge(args: argparse.Namespace) -> int:
    confirmation = (
        args.confirm
        or input(
            f"Для безвозвратного удаления медиа повторите sample_id {args.sample_id}: "
        ).strip()
    )
    if confirmation != args.sample_id:
        raise ValueError("Confirmation does not match sample_id; nothing was deleted")
    with DatasetProject(Path(args.project_root)) as project:
        removed, related_duplicates = project.purge(args.sample_id)
        print(f"Удалено локальных файлов: {len(removed)}")
        print("В приватной базе оставлен минимальный tombstone против повторного импорта.")
        if related_duplicates:
            print(
                f"В наборе есть ещё {related_duplicates} записей с тем же аудио; "
                "проверьте их согласие отдельно."
            )
    return 0


def command_video_notes_policy(args: argparse.Namespace) -> int:
    enabled = args.policy == "include"
    with DatasetProject(Path(args.project_root)) as project:
        changed = project.set_video_notes_policy(enabled)
        run_id = project.store.latest_run_id()
        if run_id:
            labels_path = project.paths.labels / f"speaker_labels_{run_id}.csv"
            write_labels_template(
                project.store,
                labels_path,
                run_id,
                speaker_aliases_path=project.paths.private / "speaker_aliases.csv",
            )
            save_report(
                project.store,
                project.paths.reports,
                run_id,
                pilot=project.store.is_first_run(run_id),
            )
        action = "возвращены" if enabled else "исключены"
        print(f"Кружки {action} из обучающей выборки: {changed}")
        print("Исходные и производные файлы не удалялись.")
    return 0


def command_reassess_quality(args: argparse.Namespace) -> int:
    with DatasetProject(Path(args.project_root)) as project:
        counts = project.reassess_quality()
        run_id = project.store.latest_run_id()
        if run_id:
            labels_path = project.paths.labels / f"speaker_labels_{run_id}.csv"
            write_labels_template(
                project.store,
                labels_path,
                run_id,
                speaker_aliases_path=project.paths.private / "speaker_aliases.csv",
            )
            save_report(
                project.store,
                project.paths.reports,
                run_id,
                pilot=project.store.is_first_run(run_id),
            )
            print(f"Таблица проверки говорящих: {labels_path}")
        print(
            "Оценка качества: "
            f"accept={counts['accept']}, review={counts['review']}, "
            f"reject={counts['reject']}, errors={counts['errors']}"
        )
        print(
            "Только голосовые: "
            f"accept={counts['voice_accept']}, review={counts['voice_review']}, "
            f"reject={counts['voice_reject']}"
        )
    return 0


def _add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", help="Начальная дата YYYY-MM-DD")
    parser.add_argument("--end", help="Конечная дата YYYY-MM-DD включительно")
    parser.add_argument("--timezone", default="Europe/Warsaw")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--expected-chat-count",
        type=int,
        help="Требовать ровно столько выбранных чатов; по умолчанию число не фиксируется",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Не скачивать записи короче этого числа секунд",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=45.0,
        help="Не скачивать записи длиннее этого числа секунд",
    )
    parser.add_argument("--order", choices=("newest", "oldest"), default="newest")
    parser.add_argument(
        "--sampling",
        choices=("balanced", "global"),
        default="balanced",
        help="Balanced alternates between chat/sender buckets; global uses time order",
    )
    parser.add_argument(
        "--next-batch",
        action="store_true",
        help="После завершённого пилота взять следующую ещё не обработанную партию",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Повторить записи с ошибками вместо перехода к следующей партии",
    )
    parser.add_argument(
        "--include-video-notes",
        action="store_true",
        help="Также собирать видеокружки; по умолчанию собираются только голосовые",
    )
    parser.add_argument("--chat-index", action="append", help="Номер/диапазон чатов")
    parser.add_argument(
        "--chat-ref",
        action="append",
        help="Стабильный псевдоним чата из предыдущего списка (рекомендуется повторно)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Не спрашивать повторное подтверждение чатов"
    )
    parser.add_argument("--language", default="ru")
    parser.add_argument(
        "--consent-status",
        choices=("confirmed_self", "confirmed_other", "unknown"),
    )
    parser.add_argument(
        "--consent-ref",
        help="Непрозрачный код локальной записи о согласии; не имя человека",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-dataset",
        description="Локальная подготовка Telegram-датасета проверки голоса",
    )
    parser.add_argument("--project-root", default=".", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Проверить локальные зависимости")
    doctor.set_defaults(handler=command_doctor)

    rank_chats = subparsers.add_parser(
        "rank-chats",
        help="Посчитать голосовые в личных чатах без скачивания медиа",
    )
    rank_chats.add_argument("--start", required=True, help="Начальная дата YYYY-MM-DD")
    rank_chats.add_argument("--end", required=True, help="Конечная дата YYYY-MM-DD включительно")
    rank_chats.add_argument("--timezone", default="Europe/Warsaw")
    rank_chats.add_argument("--top", type=int, default=20)
    rank_chats.add_argument("--min-duration", type=float, default=2.0)
    rank_chats.add_argument("--max-duration", type=float, default=45.0)
    rank_chats.set_defaults(handler=command_rank_chats)

    telegram = subparsers.add_parser(
        "collect-telegram", help="Собрать данные напрямую из выбранных Telegram-чатов"
    )
    _add_collection_arguments(telegram)
    telegram.set_defaults(handler=command_collect_telegram)

    export = subparsers.add_parser(
        "collect-export", help="Обработать локальный JSON-экспорт Telegram Desktop"
    )
    export.add_argument("--export-dir", required=True)
    _add_collection_arguments(export)
    export.set_defaults(handler=command_collect_export)

    report = subparsers.add_parser("report", help="Пересобрать отчёт запуска")
    report.add_argument("--run-id")
    report.set_defaults(handler=command_report)

    labels = subparsers.add_parser(
        "export-labels", help="Создать таблицу ручной проверки говорящих"
    )
    labels.add_argument("--run-id")
    labels.set_defaults(handler=command_export_labels)

    apply_labels = subparsers.add_parser(
        "apply-labels", help="Применить таблицу проверенных говорящих и ролей"
    )
    apply_labels.add_argument("labels")
    apply_labels.set_defaults(handler=command_apply_labels)

    split = subparsers.add_parser(
        "split", help="Создать group-safe train/validation/test разбиение"
    )
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train", type=float, default=0.8)
    split.add_argument("--validation", type=float, default=0.1)
    split.add_argument("--test", type=float, default=0.1)
    split.set_defaults(handler=command_split)

    trials = subparsers.add_parser(
        "build-trials",
        help="Создать сбалансированные пары genuine/impostor для оценки",
    )
    trials.add_argument("--seed", type=int, default=42)
    trials.set_defaults(handler=command_build_trials)

    purge = subparsers.add_parser("purge", help="Удалить медиа одной записи после отзыва согласия")
    purge.add_argument("sample_id")
    purge.add_argument("--confirm", help="Повторите точный sample_id")
    purge.set_defaults(handler=command_purge)

    video_policy = subparsers.add_parser(
        "video-notes-policy",
        help="Обратимо исключить или вернуть уже собранные кружки",
    )
    video_policy.add_argument("policy", choices=("exclude", "include"))
    video_policy.set_defaults(handler=command_video_notes_policy)

    reassess = subparsers.add_parser(
        "reassess-quality",
        help="Пересчитать качество всех WAV для speaker verification",
    )
    reassess.set_defaults(handler=command_reassess_quality)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = int(args.handler(args))
    except KeyboardInterrupt:
        print("Остановлено пользователем; уже обработанные записи сохранены.", file=sys.stderr)
        result = 130
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        result = 1
    raise SystemExit(result)
