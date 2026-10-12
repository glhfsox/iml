"""Raw recordings -> a dataset that satisfies DATA_CONTRACT.md (``{train,val,test}.csv``).

Pipeline (``build``):

1. **Decode** every source to 16 kHz mono int16. A 16 kHz mono PCM16 WAV (the Telegram collector's
   normalized output) is read as is; anything else is decoded with FFmpeg (PyAV) and resampled with
   libswresample defaults, the collector's recipe. Loudness is never changed.
2. **VAD** (the collector's): 20 ms frames; a frame is speech if its level is above
   max(-45 dBFS, p20 + 8 dB), p20 over the whole recording. Leading/trailing non-speech is trimmed
   on the frame grid.
3. **Split**: per speaker, whole recordings go to train/val/test (~70/15/15 by trimmed duration);
   a speaker with fewer than 3 recordings goes to train only. Clips of one recording never cross
   splits.
4. **Clips**: 3 s windows, hop 1.5 s in train and 3 s in val/test, tail dropped; a window is kept
   only if >= 60 % of its frames are speech.
"""

from __future__ import annotations

import json
import posixpath
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import soundfile as sf

from iml.data.contract import CONTRACT_VERSION, SPLITS, file_sha256

SR = 16000
FRAME = 320  # 20 ms
CLIP_FRAMES = 150  # 3 s
HOP_FRAMES = {"train": 75, "val": 150, "test": 150}
MIN_SPEECH_RATIO = 0.60
VAD_FLOOR_DB, VAD_OFFSET_DB = -45.0, 8.0
SPLIT_SHARE = {"val": 0.15, "test": 0.15}  # train gets the rest
MIN_RECORDINGS_TO_SPLIT = 3
RECORDINGS_DIR = "recordings"
SOURCE_COLUMNS = ["path", "speaker", "allowed"]


# --- decoding -------------------------------------------------------------------------------------


def load_pcm(path: str | Path) -> np.ndarray:
    """Any audio file -> int16 mono 16 kHz. 16 kHz mono PCM16 WAVs are read untouched."""
    path = Path(path)
    try:
        info = sf.info(str(path))
        if info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16":
            return sf.read(str(path), dtype="int16")[0]
    except sf.LibsndfileError:
        pass  # not a soundfile format (e.g. .m4a): decode with FFmpeg
    return _decode_ffmpeg(path)


def _decode_ffmpeg(path: Path) -> np.ndarray:
    import av  # optional dependency group "data"

    out = []
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for frame in c.decode(c.streams.audio[0]):
            out += [f.to_ndarray().reshape(-1) for f in rs.resample(frame)]
        out += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]  # flush
    return np.concatenate(out).astype(np.int16) if out else np.zeros(0, np.int16)


# --- VAD ------------------------------------------------------------------------------------------


def frame_db(pcm: np.ndarray) -> np.ndarray:
    """Level of each 20 ms frame in dBFS (the last partial frame over its own samples)."""
    x = np.asarray(pcm, dtype=np.float64) / 32768.0
    return np.array(
        [
            20 * np.log10(np.sqrt(np.mean(x[i : i + FRAME] ** 2)) + 1e-12)
            for i in range(0, len(x), FRAME)
        ]
    )


def speech_frames(pcm: np.ndarray) -> np.ndarray:
    """Boolean speech mask over the *full* 20 ms frames of the recording (collector VAD)."""
    db = frame_db(pcm)
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    p20 = np.sort(db)[int(np.floor(0.2 * (len(db) - 1)))]  # numpy percentile, method="lower"
    active = db > max(VAD_FLOOR_DB, p20 + VAD_OFFSET_DB)
    return active[: len(pcm) // FRAME]


def clip_starts(active: np.ndarray, hop_frames: int) -> list[int]:
    """Start frames of the kept 3 s windows: trimmed to the first..last speech frame, tail dropped,
    each window with speech ratio >= MIN_SPEECH_RATIO."""
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    first, end = int(idx[0]), int(idx[-1]) + 1
    return [
        k
        for k in range(first, end - CLIP_FRAMES + 1, hop_frames)
        if active[k : k + CLIP_FRAMES].mean() >= MIN_SPEECH_RATIO
    ]


def trimmed_seconds(active: np.ndarray) -> float:
    idx = np.flatnonzero(active)
    return 0.0 if len(idx) == 0 else (idx[-1] + 1 - idx[0]) * FRAME / SR


# --- splits ---------------------------------------------------------------------------------------


def assign_splits(recs: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Split per speaker by whole recordings.

    ``recs`` has ``recording_id, speaker, seconds`` (speech duration) and optionally ``eval_ok``
    (the recording yields at least one non-overlapping clip; only those can go to val/test).
    Recordings are sorted by id, then shuffled with ``seed``. Speakers with < 3 recordings go to
    train. Otherwise val, then test, first take the eligible recording closest to 15 % of the
    speaker's speech, then add more (in shuffled order) while that brings them closer to 15 %.
    At least one recording always stays in train.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=recs.index, dtype=object)
    eval_ok = recs["eval_ok"] if "eval_ok" in recs else pd.Series(True, index=recs.index)
    for _, g in sorted(recs.groupby("speaker"), key=lambda kv: kv[0]):
        g = g.sort_values("recording_id")
        if len(g) < MIN_RECORDINGS_TO_SPLIT:
            continue
        order = list(g.index[rng.permutation(len(g))])
        total = float(g.seconds.sum())
        for split, share in SPLIT_SHARE.items():
            target = share * total
            cand = [i for i in order if eval_ok[i]]
            if not cand or len(order) < 2:
                continue
            first = min(cand, key=lambda i: abs(float(g.seconds[i]) - target))
            out[first], got = split, float(g.seconds[first])
            order.remove(first)
            for i in [i for i in order if eval_ok[i]]:
                d = float(g.seconds[i])
                if len(order) > 1 and abs(got + d - target) < abs(got - target):
                    out[i], got = split, got + d
                    order.remove(i)
    return out


# --- build ----------------------------------------------------------------------------------------


@dataclass
class _Recording:
    recording_id: str
    path: Path
    speaker: str
    allowed: int
    pcm: np.ndarray = field(repr=False)
    active: np.ndarray = field(repr=False)


def _recording_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")


def read_sources(sources_csv: str | Path) -> pd.DataFrame:
    """The labelled raw recordings: ``sources.csv`` with ``path,speaker,allowed``, next to a
    ``recordings/`` folder; every ``path`` is ``recordings/<file>`` (relative, POSIX, normalized).
    The file stem (sanitized) is the recording id."""
    src = pd.read_csv(sources_csv, dtype={"path": str, "speaker": str}, keep_default_na=False)
    if list(src.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"{sources_csv}: columns must be {SOURCE_COLUMNS}, got {list(src.columns)}"
        )
    bad = [
        p
        for p in src.path
        if posixpath.normpath(p) != p
        or PurePosixPath(p).parts[:1] != (RECORDINGS_DIR,)
        or ".." in PurePosixPath(p).parts
        or len(PurePosixPath(p).parts) < 2
    ]
    if bad:
        raise ValueError(f"paths must be relative '{RECORDINGS_DIR}/<file>', got {bad[:3]}")
    base = Path(sources_csv).resolve().parent
    missing_files = [p for p in src.path if not (base / p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"recordings not found next to {sources_csv}: {missing_files[:3]}")
    src["recording_id"] = [_recording_id(Path(p)) for p in src.path]
    src["path"] = [base / p for p in src.path]
    src["allowed"] = pd.to_numeric(src.allowed, errors="coerce")
    if not src.allowed.isin([0, 1]).all():
        raise ValueError("allowed must be 0 or 1")
    mixed = src.groupby("speaker").allowed.nunique()
    if (mixed > 1).any():
        raise ValueError(f"speakers with both allowed=0 and 1: {list(mixed[mixed > 1].index)}")
    dup = src.recording_id[src.recording_id.duplicated()]
    if len(dup):
        raise ValueError(f"duplicate recording_id: {dup.tolist()[:5]}")
    return src


def build(
    sources_csv: str | Path, out_dir: str | Path, seed: int = 42, overwrite: bool = False
) -> dict:
    """Build a contract dataset in ``out_dir``; returns the build report (also written as JSON).

    Everything is decoded, split and checked first and written into a temporary sibling folder;
    only a complete dataset replaces the old one (``overwrite``), so a failed build changes nothing.
    """
    out = Path(out_dir).resolve()
    targets = ["audio", *(f"{s}.csv" for s in SPLITS), "build_report.json"]
    existing = [t for t in targets if (out / t).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out} already has {existing} (pass overwrite=True to replace)")
    src = read_sources(sources_csv)
    inside = [str(p) for p in src.path if Path(p).resolve().is_relative_to(out)]
    if inside:
        raise ValueError(f"source recordings must not live inside the output folder: {inside[:3]}")

    recs = []
    for r in src.itertuples():
        pcm = load_pcm(r.path)
        recs.append(
            _Recording(r.recording_id, r.path, r.speaker, int(r.allowed), pcm, speech_frames(pcm))
        )
    table = pd.DataFrame(
        {
            "recording_id": [r.recording_id for r in recs],
            "speaker": [r.speaker for r in recs],
            "seconds": [r.active.sum() * FRAME / SR for r in recs],
            "eval_ok": [len(clip_starts(r.active, HOP_FRAMES["val"])) > 0 for r in recs],
            "usable": [len(clip_starts(r.active, HOP_FRAMES["train"])) > 0 for r in recs],
        }
    )
    table["split"] = None
    usable = table[table.usable]
    table.loc[usable.index, "split"] = assign_splits(usable, seed)
    starts = [
        clip_starts(r.active, HOP_FRAMES[s]) if s else []
        for r, s in zip(recs, table.split, strict=True)
    ]
    for split in SPLITS:
        labels = {
            r.allowed for r, s, k in zip(recs, table.split, starts, strict=True) if s == split and k
        }
        if labels != {0, 1}:
            raise ValueError(
                f"split {split!r} would have labels {sorted(labels)}; both classes are required"
            )

    tmp = out.parent / f".{out.name}.building"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "audio").mkdir(parents=True)
    rows = {s: [] for s in SPLITS}
    report_recs = []
    for rec, split, ks in zip(recs, table.split, starts, strict=True):
        for k in ks:
            rel = f"audio/{rec.recording_id}_{k * FRAME * 1000 // SR:06d}.wav"
            clip = rec.pcm[k * FRAME : (k + CLIP_FRAMES) * FRAME]
            sf.write(str(tmp / rel), clip, SR, subtype="PCM_16")
            rows[split].append((rel, rec.allowed))
        report_recs.append(
            dict(
                recording_id=rec.recording_id,
                source=str(rec.path),
                sha256=file_sha256(rec.path),
                speaker=rec.speaker,
                allowed=rec.allowed,
                split=split,
                seconds=round(len(rec.pcm) / SR, 3),
                speech_seconds=round(rec.active.sum() * FRAME / SR, 3),
                trimmed_seconds=round(trimmed_seconds(rec.active), 3),
                clips=len(ks),
                dropped=None if ks else "no 3 s window with >= 60 % speech",
            )
        )
    for split in SPLITS:
        pd.DataFrame(sorted(rows[split]), columns=["path", "allowed"]).to_csv(
            tmp / f"{split}.csv", index=False
        )

    rec_df = pd.DataFrame(report_recs)
    clips = pd.DataFrame([(s, a) for s in SPLITS for _, a in rows[s]], columns=["split", "allowed"])
    report = dict(
        contract_version=CONTRACT_VERSION,
        seed=seed,
        sources=str(Path(sources_csv)),
        params=dict(
            sample_rate=SR,
            clip_s=CLIP_FRAMES * FRAME / SR,
            hop_s={k: v * FRAME / SR for k, v in HOP_FRAMES.items()},
            min_speech_ratio=MIN_SPEECH_RATIO,
            split_share=SPLIT_SHARE,
        ),
        clips=clips.groupby(["split", "allowed"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index"),
        recordings=report_recs,
        dropped=rec_df[rec_df.clips == 0][["recording_id", "speaker", "dropped"]].to_dict(
            orient="records"
        ),
    )
    (tmp / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=int)
    )

    # swap the finished dataset in: old targets -> backup, new -> out, drop the backup
    out.mkdir(parents=True, exist_ok=True)
    backup = out.parent / f".{out.name}.previous"
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir()
    for t in existing:
        (out / t).rename(backup / t)
    for t in targets:
        (tmp / t).rename(out / t)
    shutil.rmtree(backup)
    shutil.rmtree(tmp)
    return report
