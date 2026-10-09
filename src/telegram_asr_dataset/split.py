from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from .state import StateStore


def assign_splits(
    store: StateStore,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, int]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be non-negative and sum to 1")

    groups: dict[str, list[Any]] = defaultdict(list)
    for row in store.rows():
        if row["dataset_status"] == "ready" and row["dedupe_status"] == "unique":
            groups[str(row["split_group_id"])].append(row)
    if len(groups) < 3:
        raise ValueError(
            "At least three ready, independent groups are required for train/validation/test"
        )

    groups_by_speaker: dict[str, list[str]] = defaultdict(list)
    for group, rows in groups.items():
        speakers = {str(row["speaker_ref"]) for row in rows}
        if len(speakers) != 1 or "" in speakers:
            raise ValueError("Every ready split group must contain exactly one verified speaker")
        groups_by_speaker[next(iter(speakers))].append(group)

    assignments: dict[str, str] = {}
    for speaker, speaker_groups in sorted(groups_by_speaker.items()):
        if len(speaker_groups) < 3:
            raise ValueError(f"Speaker {speaker} needs at least three independent sessions/groups")
        ordered_groups = sorted(
            speaker_groups,
            key=lambda group: hashlib.sha256(f"{seed}:{speaker}:{group}".encode()).hexdigest(),
        )
        total = len(ordered_groups)
        train_count = max(1, round(total * train_ratio)) if train_ratio else 0
        validation_count = max(1, round(total * validation_ratio)) if validation_ratio else 0
        test_count = total - train_count - validation_count
        if test_ratio and test_count < 1:
            test_count = 1
            if train_count >= validation_count and train_count > 1:
                train_count -= 1
            elif validation_count > 1:
                validation_count -= 1
        while train_count + validation_count + test_count > total:
            if train_count > 1:
                train_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                test_count -= 1
        for index, group in enumerate(ordered_groups):
            if index < train_count:
                assignments[group] = "train"
            elif index < train_count + validation_count:
                assignments[group] = "validation"
            else:
                assignments[group] = "test"

    version = (
        f"v2_session_stratified_seed={seed}_ratios="
        f"{train_ratio:.6f},{validation_ratio:.6f},{test_ratio:.6f}"
    )
    counts = {"train": 0, "validation": 0, "test": 0}
    updates: list[tuple[str, dict[str, str]]] = []
    for group, rows in groups.items():
        split_name = assignments[group]
        for row in rows:
            updates.append(
                (
                    str(row["sample_id"]),
                    {"split": split_name, "split_version": version},
                )
            )
            counts[split_name] += 1
    store.update_many(updates)
    return counts


def assert_group_safe(store: StateStore) -> None:
    seen: dict[str, str] = {}
    for row in store.rows():
        split_name = str(row["split"])
        if split_name not in {"train", "validation", "test"}:
            continue
        group = str(row["split_group_id"])
        prior = seen.setdefault(group, split_name)
        if prior != split_name:
            raise AssertionError(f"Split leakage: group {group} occurs in {prior} and {split_name}")
