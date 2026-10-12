# Data contract (v1)

Every dataset the team trains or evaluates on has this shape. Whoever produces a dataset guarantees
everything below; whoever consumes it can rely on it and needs nothing else.

## Layout

```
<dataset>/
  train.csv
  val.csv
  test.csv
  audio/...wav          # any layout under the folder; referenced from the CSVs
  build_report.json     # optional, informational only (how it was built); not part of the contract
```

Each CSV has exactly two columns, in this order:

| column    | meaning |
|-----------|---------|
| `path`    | clip path, relative to the dataset folder, POSIX (`/`), normalized (no `..`, `./`, `//`) |
| `allowed` | `1` = this person may enter, `0` = must be rejected |

```csv
path,allowed
audio/rec_0929b045510095ce_000480.wav,1
audio/S_british_001200.wav,0
```

## Guarantees

**Clips**
- WAV, PCM 16-bit, 16 kHz, mono, **exactly 48 000 samples (3.000 s)**.
- Loudness is the source's: no gain or loudness normalization.
- At least **60 % speech** per clip. Speech = 20 ms frames whose level exceeds max(−45 dBFS, p20 + 8 dB),
  with p20 the 20th percentile of frame levels over the whole source recording. Silence before the first
  and after the last speech frame of a recording is trimmed. The producer guarantees this; it can't be
  re-measured on a 3 s clip alone, because the threshold depends on the whole recording.

**Splits**
- Each `path` exists, is unique and appears in exactly one split.
- All clips cut from one source recording are in the same split (no clip-level leakage). Splits are by
  recording, not by session/day: different recordings of the same day may land in different splits.
- Each speaker's recordings are spread over train/val/test at about 70/15/15 of speech duration;
  val and test only get recordings that yield at least one clip, and at least one recording stays in
  train. A speaker with fewer than 3 usable recordings goes to train only. Recordings that yield no
  clip are dropped (listed in `build_report.json`).
- Train clips overlap by 50 % (hop 1.5 s). Val/test clips don't overlap (hop 3 s).
- Train, val and test each contain both classes.

## Producing and checking a dataset

```bash
uv sync --group dev --group data                                  # PyAV/ffmpeg for decoding raw audio
uv run iml-data build --sources <raw>/sources.csv --out <dataset> [--seed 42] [--overwrite]
uv run iml-data validate <dataset>                                # exits 1 and lists violations
```

Raw input is a folder with the recordings and, right next to them, the CSV that labels them:

```
<raw>/
  sources.csv           # path,speaker,allowed
  recordings/           # one file per recording, any ffmpeg-readable format (.wav, .oga, .m4a, ...)
```

```csv
path,speaker,allowed
recordings/rec_0929b045510095ce.wav,M,1
recordings/S_british.m4a,S,0
```

- `path` is relative to `sources.csv`: always `recordings/<file>` (POSIX, no `..`, no absolute paths).
- The file name (without extension) is the recording id, so it must be unique; clip names start with it.
- `allowed` is `1`/`0` and must be the same for all recordings of a speaker.

The pipeline lives in `src/iml/data/preprocess.py`:
1. decode to 16 kHz mono int16, the Telegram collector's recipe;
2. VAD and trimming;
3. per-speaker split by whole recordings;
4. 3 s windows with the speech threshold.

A build is atomic: it is written to a temporary folder and only a complete, valid dataset replaces the
old one, so a failed `--overwrite` leaves the previous dataset intact. Raw recordings must not live
inside the output folder. The validator lives in `src/iml/data/contract.py`.

## Using a dataset

```bash
uv run iml-train --config configs/cnn10.json --data <dataset> --member <you> --topic <topic>
export IML_DATA=<dataset>                                         # or once per shell
```

- `load_splits` validates the folder and builds one `SpectrogramDataset` per split.
- `ExperimentConfig.excerpt_sec` defaults to 3.0, the clip length, so nothing is cropped or padded.
  Configs generated before this change still say 2.0: re-run `make configs`.
- Each run writes the dataset fingerprint (manifests + clip bytes) to `data.json`. A finished run is
  only reused for the same dataset; for other data, or for runs from before data tracking (no
  `data.json`), `iml-train`/`iml-sweep` stop and ask for `--overwrite`.
- In notebooks: `load_splits(cfg, data_dir=DATA)`, and `check_run_data` / `record_run_data` from
  `iml.cli._data` for the same guard (see `notebooks/_template`).

## Changing the contract

Any change to the guarantees above bumps `CONTRACT_VERSION` in `src/iml/data/contract.py` and the
version in this file's title. Datasets record the version they were built with in `build_report.json`.

## Known limitations of the first dataset (glhfsox v1)

- The only `allowed = 0` speaker (S) was recorded on a different phone and codec (AAC `.m4a`) than the
  allowed speakers (Telegram Opus voice notes). A model can learn the **channel** instead of the voice.
  Add negatives recorded the same way as positives before trusting FAR.
- With a single negative speaker, the test set does not measure rejection of unseen people.
