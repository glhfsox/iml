from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

from .state import StateStore

TRIAL_FIELDS = [
    "trial_id",
    "split",
    "enrollment_sample_id",
    "enrollment_audio_path",
    "test_sample_id",
    "test_audio_path",
    "claimed_speaker_ref",
    "actual_speaker_ref",
    "expected_same_speaker",
]


def _trial_id(first: Any, second: Any, split: str) -> str:
    value = f"{split}|{first['sample_id']}|{second['sample_id']}"
    return "trial_" + hashlib.sha256(value.encode()).hexdigest()[:16]


def build_verification_trials(
    store: StateStore,
    destination: Path,
    splits: tuple[str, ...] = ("validation", "test"),
    seed: int = 42,
) -> dict[str, int]:
    rows = [
        row
        for row in store.rows()
        if row["dataset_status"] == "ready"
        and row["split"] in splits
        and row["speaker_status"] == "verified_single"
    ]
    output: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        speakers = sorted({str(row["speaker_ref"]) for row in split_rows})
        if len(speakers) < 2:
            raise ValueError(f"Split {split} needs at least two speakers for trials")

        genuine: list[tuple[Any, Any]] = []
        impostor: list[tuple[Any, Any]] = []
        for first, second in combinations(split_rows, 2):
            same_speaker = first["speaker_ref"] == second["speaker_ref"]
            if same_speaker and first["session_ref"] == second["session_ref"]:
                continue
            (genuine if same_speaker else impostor).append((first, second))
        if not genuine:
            raise ValueError(f"Split {split} needs genuine pairs from different sessions")
        impostor.sort(
            key=lambda pair: hashlib.sha256(
                f"{seed}:{split}:{pair[0]['sample_id']}:{pair[1]['sample_id']}".encode()
            ).hexdigest()
        )
        selected_pairs = [
            *((first, second, True) for first, second in genuine),
            *((first, second, False) for first, second in impostor[: len(genuine)]),
        ]
        counts[split] = len(selected_pairs)
        for first, second, same_speaker in selected_pairs:
            output.append(
                {
                    "trial_id": _trial_id(first, second, split),
                    "split": split,
                    "enrollment_sample_id": first["sample_id"],
                    "enrollment_audio_path": first["normalized_audio_path"],
                    "test_sample_id": second["sample_id"],
                    "test_audio_path": second["normalized_audio_path"],
                    "claimed_speaker_ref": first["speaker_ref"],
                    "actual_speaker_ref": second["speaker_ref"],
                    "expected_same_speaker": int(same_speaker),
                }
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(output)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)
    try:
        os.chmod(destination, 0o600)
    except OSError:
        pass
    return counts
