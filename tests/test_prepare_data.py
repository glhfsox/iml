import csv
import hashlib
import json
import wave

import numpy as np
import pytest
import soundfile as sf

from iml.cli._data import load_splits
from iml.cli.train import main as train_main
from iml.config import ExperimentConfig
from iml.data.manifest import load_split_manifest, prepare_dataset

FIELDS = [
    "sample_id",
    "source_recording_id",
    "media_type",
    "forwarded",
    "normalized_audio_path",
    "audio_pcm_sha256",
    "speaker_ref",
    "speaker_status",
    "session_ref",
    "consent_status",
    "manual_review_status",
    "quality_decision",
    "dedupe_status",
    "dataset_status",
]


def _pcm_sha256(path):
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as source:
        for frames in iter(lambda: source.readframes(65_536), b""):
            digest.update(frames)
    return digest.hexdigest()


def _reviewed_dataset(tmp_path, duplicate_audio=False):
    rows = []
    audio = tmp_path / "collector" / "data" / "normalized"
    audio.mkdir(parents=True)
    speakers = {
        "speaker_01": ("day_01", "day_02", "day_03"),
        "speaker_02": ("day_01", "day_02", "day_03"),
        "speaker_03": ("day_01", "day_02", "day_03"),
        "speaker_04": ("day_01", "day_02", "day_03"),
    }
    index = 0
    for speaker, sessions in speakers.items():
        for session in sessions:
            for _take in range(2):
                index += 1
                sample_id = f"rec_{index:03d}"
                path = audio / f"{sample_id}.wav"
                time = np.arange(3 * 16_000) / 16_000
                frequency = 150 if duplicate_audio else 150 + index * 7
                waveform = 0.1 * np.sin(2 * np.pi * frequency * time)
                sf.write(path, waveform.astype(np.float32), 16_000, subtype="PCM_16")
                digest = _pcm_sha256(path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source_recording_id": sample_id,
                        "media_type": "voice",
                        "forwarded": "0",
                        "normalized_audio_path": f"data/normalized/{path.name}",
                        "audio_pcm_sha256": digest,
                        "speaker_ref": speaker,
                        "speaker_status": "verified_single",
                        "session_ref": f"{speaker}_{session}",
                        "consent_status": (
                            "confirmed_self" if speaker == "speaker_01" else "confirmed_other"
                        ),
                        "manual_review_status": "approved",
                        "quality_decision": "accept",
                        "dedupe_status": "unique",
                        "dataset_status": "ready",
                    }
                )
    rows.append(
        {
            **rows[0],
            "sample_id": "pending_sample",
            "source_recording_id": "pending_sample",
            "manual_review_status": "pending",
            "dataset_status": "pending_review",
        }
    )
    metadata = tmp_path / "collector" / "data" / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata, tmp_path / "collector"


def test_prepare_dataset_has_no_session_leakage(tmp_path):
    metadata, root = _reviewed_dataset(tmp_path)
    output = tmp_path / "processed"
    report = prepare_dataset(metadata, root, output, "speaker_01")

    assert report["approved_recordings"] == 24
    assert report["excluded_rows"] == {"pending_manual_review": 1}
    assert all(
        values == {"total": 8, "target": 2, "impostor": 6} for values in report["counts"].values()
    )

    payload = load_split_manifest(output / "splits.json")
    assert payload["dataset_id"] == report["dataset_id"]
    raw = json.loads((output / "splits.json").read_text())
    speaker_sessions = {}
    for split, items in raw["splits"].items():
        for item in items:
            assert (output / item["path"]).is_file()
            key = (item["speaker_ref"], item["session_ref"])
            assert speaker_sessions.setdefault(key, split) == split


def test_real_manifest_builds_spectrogram_loaders(tmp_path):
    metadata, root = _reviewed_dataset(tmp_path)
    output = tmp_path / "processed"
    report = prepare_dataset(metadata, root, output, "speaker_01")
    cfg = ExperimentConfig(
        dataset_id=report["dataset_id"], excerpt_sec=1.0, batch_size=2, num_workers=0
    )
    loaders = load_splits(cfg, manifest_path=str(output / "splits.json"))
    x, y = next(iter(loaders["train"]))
    assert x.shape[:3] == (2, 1, cfg.n_mels)
    assert set(y.tolist()) <= {0, 1}


def test_duplicate_reviewed_pcm_is_rejected(tmp_path):
    metadata, root = _reviewed_dataset(tmp_path, duplicate_audio=True)
    with pytest.raises(ValueError, match="duplicate PCM"):
        prepare_dataset(metadata, root, tmp_path / "processed", "speaker_01")


def test_manifest_tampering_is_rejected(tmp_path):
    metadata, root = _reviewed_dataset(tmp_path)
    output = tmp_path / "processed"
    prepare_dataset(metadata, root, output, "speaker_01")
    path = output / "splits.json"
    payload = json.loads(path.read_text())
    payload["splits"]["train"][0]["pcm_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch|fingerprint"):
        load_split_manifest(path)


def test_train_cli_records_the_exact_real_data_split(tmp_path):
    metadata, root = _reviewed_dataset(tmp_path)
    output = tmp_path / "processed"
    prepare_dataset(metadata, root, output, "speaker_01")
    config = ExperimentConfig(
        width=4,
        epochs=1,
        batch_size=4,
        num_workers=0,
        member="test",
        topic="real_manifest",
    )
    config_path = tmp_path / "config.json"
    config.to_json(config_path)
    runs = tmp_path / "runs"
    train_main(
        [
            "--config",
            str(config_path),
            "--data-manifest",
            str(output / "splits.json"),
            "--runs-root",
            str(runs),
            "--device",
            "cpu",
        ]
    )
    recorded = list(runs.rglob("splits.json"))
    assert len(recorded) == 1
    assert recorded[0].read_bytes() == (output / "splits.json").read_bytes()
