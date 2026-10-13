"""Data contract (iml.data.contract), the preprocessing that produces it and load_splits on top."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from iml.cli._data import check_run_data, load_splits
from iml.config import ExperimentConfig
from iml.data.contract import fingerprint, validate_dataset
from iml.data.preprocess import (
    FRAME,
    assign_splits,
    build,
    clip_starts,
    read_sources,
    speech_frames,
)

SR = 16000


def _speechy(seconds: float, seed: int, pause_every: float = 0.0) -> np.ndarray:
    """Speech-like noise: 0.26 s "syllables" with 0.1 s quiet gaps (so the collector VAD's p20 floor
    sits on the gaps), a 0.5 s silent lead-in, and optional 0.4 s pauses every `pause_every` s."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.1, int(seconds * SR))
    gap = (np.arange(len(x)) % int(0.36 * SR)) >= int(0.26 * SR)
    x[gap] *= 0.01
    if pause_every:
        for t in np.arange(pause_every, seconds, pause_every):
            x[int(t * SR) : int((t + 0.4) * SR)] *= 0.001
    x[: int(0.5 * SR)] *= 0.001  # leading silence to trim
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def _write_dataset(d, rows: dict[str, list[tuple[str, int]]]):
    (d / "audio").mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        for i, (name, _) in enumerate(items):
            sf.write(d / "audio" / name, _speechy(3.5, i)[-48000:], SR, subtype="PCM_16")
        pd.DataFrame(items, columns=["path", "allowed"]).assign(
            path=lambda x: "audio/" + x.path
        ).to_csv(d / f"{split}.csv", index=False)


@pytest.fixture
def tiny(tmp_path):
    rows = {s: [(f"{s}{k}.wav", k % 2) for k in range(4)] for s in ("train", "val", "test")}
    _write_dataset(tmp_path, rows)
    return tmp_path


def test_valid_dataset_passes(tiny):
    assert validate_dataset(tiny) == []
    assert fingerprint(tiny)["counts"]["train"] == {"allowed": 2, "not_allowed": 2}


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_dataset_is_valid(tmp_path, label):
    """Contract v2: a dataset (and every split) may hold one class, e.g. an enrollment bank."""
    rows = {s: [(f"{s}{k}.wav", label) for k in range(3)] for s in ("train", "val", "test")}
    _write_dataset(tmp_path, rows)
    assert validate_dataset(tmp_path) == []
    counts = fingerprint(tmp_path)["counts"]
    assert counts["val"] == {"allowed": 3 * label, "not_allowed": 3 * (1 - label)}


def test_fingerprint_hash_unchanged_by_v2():
    """The v2 bump must not change dataset sha1s, or every finished run would need retraining."""
    from iml.data import contract

    assert contract.CONTRACT_VERSION == 2 and contract.FINGERPRINT_SALT == "contract1"


@pytest.mark.parametrize(
    "breakage, expected",
    [
        (lambda d: (d / "val.csv").write_text("file,label\naudio/val0.wav,1\n"), "header"),
        (
            lambda d: (d / "val.csv").write_text(
                "path,allowed\naudio/val0.wav,2\naudio/val1.wav,0\n"
            ),
            "0 or 1",
        ),
        (
            lambda d: (d / "test.csv").write_text(
                "path,allowed\naudio/val0.wav,1\naudio/test1.wav,0\n"
            ),
            "already listed",
        ),
        (
            lambda d: (d / "test.csv").write_text("path,allowed\n../x.wav,1\naudio/test1.wav,0\n"),
            "relative POSIX",
        ),
        (
            lambda d: sf.write(
                d / "audio/train0.wav", np.zeros(32000, np.int16) + 3000, SR, subtype="PCM_16"
            ),
            "exactly 48000",
        ),
        (
            lambda d: sf.write(
                d / "audio/train0.wav", np.zeros(48000, np.int16) + 3000, 8000, subtype="PCM_16"
            ),
            "sample rate",
        ),
        (lambda d: (d / "val.csv").write_text(""), "header"),
        (
            lambda d: (d / "test.csv").write_text(
                "path,allowed\naudio//val0.wav,1\naudio/test1.wav,0\n"
            ),
            "normalized",
        ),
        (lambda d: (d / "train.csv").unlink(), "missing"),
    ],
)
def test_violations_are_reported(tiny, breakage, expected):
    breakage(tiny)
    errors = validate_dataset(tiny)
    assert any(expected in e for e in errors), errors


def test_clips_are_trimmed_and_speech_filtered():
    pcm = _speechy(10.0, 0)
    active = speech_frames(pcm)
    starts = clip_starts(active, hop_frames=150)
    assert starts and starts[0] * FRAME >= int(0.5 * SR) - FRAME  # leading silence trimmed
    assert all(active[k : k + 150].mean() >= 0.6 for k in starts)
    quiet = np.zeros(10 * SR, np.int16)
    assert clip_starts(speech_frames(quiet), 75) == []


def test_assign_splits_rules():
    recs = pd.DataFrame(
        {
            "recording_id": [f"a{i}" for i in range(10)] + ["d0"],
            "speaker": ["A"] * 10 + ["D"],
            "seconds": [10.0] * 10 + [5.0],
        }
    ).sample(frac=1, random_state=0)
    s1, s2 = assign_splits(recs, seed=1), assign_splits(recs, seed=1)
    assert s1.equals(s2)  # deterministic
    a = s1[recs.speaker == "A"].value_counts()
    assert a["val"] >= 1 and a["test"] >= 1 and a["train"] >= 6
    assert s1[recs.speaker == "D"].tolist() == ["train"]  # < 3 recordings -> train only
    shuffled = recs.sample(frac=1, random_state=5)  # row order must not matter
    assert assign_splits(shuffled, seed=1).sort_index().equals(s1.sort_index())


@pytest.mark.parametrize("seed", range(10))
def test_assign_splits_only_eval_ok_recordings_go_to_val_test(seed):
    recs = pd.DataFrame(
        {
            "recording_id": ["a0", "a1", "a2", "a3", "a4"],
            "speaker": ["A"] * 5,
            "seconds": [2.5, 10.0, 10.0, 10.0, 60.0],
            "eval_ok": [False, True, True, True, True],
        }
    )
    s = assign_splits(recs, seed=seed)
    assert s[0] == "train" and (s == "val").sum() >= 1 and (s == "test").sum() >= 1
    assert s[4] == "train"  # a recording far above 15 % is never forced into val/test


def test_build_end_to_end(tmp_path):
    src = []
    for spk, allowed in (("A", 1), ("B", 0)):
        for i in range(4):
            p = tmp_path / "recordings" / f"{spk}{i}.wav"
            p.parent.mkdir(exist_ok=True)
            sf.write(
                p, _speechy(12.0, hash((spk, i)) % 1000, pause_every=4.0), SR, subtype="PCM_16"
            )
            src.append(dict(path=f"recordings/{spk}{i}.wav", speaker=spk, allowed=allowed))
    pd.DataFrame(src).to_csv(tmp_path / "sources.csv", index=False)
    report = build(tmp_path / "sources.csv", tmp_path / "ds", seed=0)
    assert validate_dataset(tmp_path / "ds") == []
    per_rec = pd.DataFrame(report["recordings"])
    assert per_rec.groupby("recording_id").split.nunique().max() == 1
    with pytest.raises(FileExistsError):
        build(tmp_path / "sources.csv", tmp_path / "ds")
    # a failing rebuild leaves the previous dataset untouched
    before = fingerprint(tmp_path / "ds")
    bad = pd.DataFrame(src)
    bad.loc[0, "allowed"] = 1 - bad.loc[0, "allowed"]  # one speaker with both labels
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="both allowed=0 and 1"):
        build(tmp_path / "bad.csv", tmp_path / "ds", overwrite=True)
    assert fingerprint(tmp_path / "ds") == before and validate_dataset(tmp_path / "ds") == []
    # raw recordings inside the output folder are refused, never deleted
    with pytest.raises(ValueError, match="inside the output folder"):
        build(tmp_path / "sources.csv", tmp_path, overwrite=True)
    assert (tmp_path / "recordings" / "A0.wav").exists()


def test_build_single_class_sources(tmp_path):
    """Contract v2: sources where every speaker is allowed=1 build into a valid dataset."""
    src = []
    for spk in ("A", "B"):
        for i in range(4):
            p = tmp_path / "recordings" / f"{spk}{i}.wav"
            p.parent.mkdir(exist_ok=True)
            sf.write(
                p, _speechy(12.0, hash((spk, i)) % 1000, pause_every=4.0), SR, subtype="PCM_16"
            )
            src.append(dict(path=f"recordings/{spk}{i}.wav", speaker=spk, allowed=1))
    pd.DataFrame(src).to_csv(tmp_path / "sources.csv", index=False)
    build(tmp_path / "sources.csv", tmp_path / "ds", seed=0)
    assert validate_dataset(tmp_path / "ds") == []
    counts = fingerprint(tmp_path / "ds")["counts"]
    assert all(c["not_allowed"] == 0 and c["allowed"] > 0 for c in counts.values()), counts


@pytest.mark.parametrize("split", ["train", "val"])
def test_load_splits_refuses_single_class_training_split(tiny, split):
    (tiny / f"{split}.csv").write_text(
        f"path,allowed\naudio/{split}1.wav,1\naudio/{split}3.wav,1\n"
    )
    assert validate_dataset(tiny) == []  # valid under the contract ...
    cfg = ExperimentConfig(batch_size=2, num_workers=0)
    with pytest.raises(ValueError, match=f"split {split} has only allowed=1; training needs both"):
        load_splits(cfg, data_dir=tiny)  # ... but not trainable


def test_iml_train_fails_early_on_single_class_dataset(tiny, tmp_path):
    from iml.cli.train import main

    for split in ("train", "val", "test"):  # an enrollment-bank-like dataset: all allowed=1
        (tiny / f"{split}.csv").write_text(
            f"path,allowed\naudio/{split}1.wav,1\naudio/{split}3.wav,1\n"
        )
    ExperimentConfig(width=4, epochs=1, batch_size=2, num_workers=0).to_json(tmp_path / "c.json")
    runs = tmp_path / "runs"
    with pytest.raises(ValueError, match="split train has only allowed=1; training needs both"):
        main(["--config", str(tmp_path / "c.json"), "--data", str(tiny), "--runs-root", str(runs)])
    assert not list(runs.rglob("metrics.json"))


def test_load_splits_warns_on_single_class_test(tiny):
    (tiny / "test.csv").write_text("path,allowed\naudio/test0.wav,0\naudio/test2.wav,0\n")
    cfg = ExperimentConfig(batch_size=2, num_workers=0)
    with pytest.warns(UserWarning, match="split test has only allowed=0"):
        loaders = load_splits(cfg, data_dir=tiny)
    assert set(loaders) == {"train", "val", "test"}


def test_load_splits_and_run_data_guard(tiny, tmp_path):
    cfg = ExperimentConfig(batch_size=2, num_workers=0)
    loaders = load_splits(cfg, data_dir=tiny)
    x, y = next(iter(loaders["val"]))
    assert x.shape[1:] == (1, cfg.n_mels, 301) and set(y.tolist()) <= {0, 1}
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text("{}")
    (run / "data.json").write_text('{"sha1": "old"}')
    with pytest.raises(RuntimeError):
        check_run_data(run, fingerprint(tiny), overwrite=False)
    check_run_data(run, fingerprint(tiny), overwrite=True)
    (run / "data.json").unlink()  # finished before data tracking: data unknown
    with pytest.raises(RuntimeError, match="no data.json"):
        check_run_data(run, fingerprint(tiny), overwrite=False)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.wav",
        "/abs/a.wav",
        "raw/a.wav",
        "recordings/../a.wav",
        "./recordings/a.wav",
        "recordings",
    ],
)
def test_sources_paths_must_be_inside_recordings(tmp_path, path):
    (tmp_path / "recordings").mkdir()
    sf.write(tmp_path / "recordings" / "a.wav", _speechy(4.0, 0), SR, subtype="PCM_16")
    pd.DataFrame([dict(path=path, speaker="A", allowed=1)]).to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(ValueError, match="recordings/<file>"):
        read_sources(tmp_path / "s.csv")
    pd.DataFrame([dict(path="recordings/a.wav", speaker="A", allowed=1)]).to_csv(
        tmp_path / "s.csv", index=False
    )
    assert read_sources(tmp_path / "s.csv").recording_id.tolist() == ["a"]
