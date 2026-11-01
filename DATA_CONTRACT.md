# Data contract (v3)

This contract describes waveform datasets consumed by the binary training commands and shared
notebook loader. The separate `sv_*` speaker-embedding datasets use a different format.

## Layout

```
<dataset>/
  train.csv
  val.csv
  test.csv
  audio/...npy          # any layout under the folder; referenced from the CSVs
```

Each CSV has exactly two columns, in this order:

| column    | meaning |
|-----------|---------|
| `path`    | clip path, relative to the dataset folder, POSIX (`/`), normalized (no `..`, `./`, `//`) |
| `allowed` | `1` = this person may enter, `0` = must be rejected |

```csv
path,allowed
audio/rec_0929b045510095ce_000480.npy,1
audio/N174_174-168635-0003_000600.npy,0
```

## Current unified training dataset

`data/training_dataset_v2` is currently the **single source of unified data for binary model
training**. Select it with `--data data/training_dataset_v2` or set
`IML_DATA=data/training_dataset_v2`. The CLI has no default dataset and never merges folders.

Counts recalculated from the CSVs on 2026-11-01:

| Speaker / class | Train clips | Val clips | Test clips | Total clips |
|---|---:|---:|---:|---:|
| M, valid (`1`) | 57 | 7 | 7 | 71 |
| P, valid (`1`) | 57 | 7 | 7 | 71 |
| MS, valid (`1`) | 57 | 7 | 7 | 71 |
| D, valid (`1`) | 57 | 7 | 7 | 71 |
| **All valid** | **228** | **28** | **28** | **284** |
| T, invalid (`0`) | 57 | 7 | 7 | 71 |
| External N speakers, invalid (`0`) | 228 | 28 | 28 | 284 |
| **All invalid** | **285** | **35** | **35** | **355** |
| **Dataset total** | **513** | **63** | **63** | **639** |

There are **71 clips per valid person**, **639 clips overall**, and **259 distinct source
recording IDs** inferred from filenames. Each clip is a stored 3-second excerpt, not an
independent original recording. The N group contains 40 speaker IDs.
**Recalculate these counts and update this document and README whenever new data arrives
or data, labels, split membership, naming, or the contract changes.** These counts are a
snapshot, not fixed contract requirements. Validate the dataset after changes.

`data/our_voices` is the broader team voice collection, including T negatives and 38 anonymous
clips whose identities remain unresolved. It is not the unified training source; enrollment
banks must use only verified allowed speakers with `allowed=1`. The `sv_clean`, `sv_nosil`
and `sv_noise` datasets use speaker-identity Parquet manifests and precomputed spectrogram
arrays, so they cannot be passed directly to the current binary training loader. I guess that currently, those Parquet files are used in a different branch (Artem one), we DO NOT use them in the main branch, as they have different manifests.

## Naming conventions in the shared datasets

Clip names produced by `iml-data build` are `<recording_id>_<start_ms>.npy`: the final
number is the start offset in the source recording, in milliseconds, padded to at least six
digits. The source filename stem is sanitized to form the recording ID. Speaker prefixes
are a convention in the shared datasets, not a required format checked by the validator.
The CSV `allowed` value determines the class; filenames do not determine it.

- `T_rec_<id>_<start_ms>.npy` identifies T, with `allowed=0` in both `training_dataset_v2`
  and `our_voices`. Only the 13 recording IDs matched to T in `training_dataset_v2` were
  renamed in `our_voices`; other `rec_*` files retain their names.
- `M_*`, `P_*`, `D_*`, and `MS_*` identify the corresponding speakers. MS was formerly
  called S. The former `MH_*` clips in `training_dataset_v2` have been renamed `MS_*`.
  All 71 were verified against the corresponding offsets in the five MS accent recordings
  in `our_voices` (waveform correlation > 0.99999999997). They are the same source audio;
  different clip boundaries and tiny sample differences can prevent exact hash matches.
- Bare `rec_<id>_<start_ms>.npy` names do not identify a speaker on their own. Of the
  remaining such clips in `our_voices`, recording-ID matches to `training_dataset_v2`
  identify 80 clips as M and 64 as P. Another 38 clips (9 recording IDs) have no speaker
  mapping in the available manifests; do not infer their identity from the `rec_` prefix.
- `N<speaker-id>_*` denotes external negative speakers in `training_dataset_v2`.

Keep the recording ID and offset when adding a speaker prefix, and update the split CSV
paths alongside the filenames. Use source metadata or a verified recording-ID mapping to
identify speakers. When copying clips between datasets, preserve their source-recording
split assignments and labels rather than randomly splitting individual clips.

`our_voices` retains 224 D clips and 296 MS clips. `training_dataset_v2` contains 71 clips
each for M, P, MS and D, all labeled `allowed=1`, plus T and external N negatives. Each of
these four valid speakers has 57 train, 7 val and 7 test clips. D was sampled deterministically
from all five source recordings: 19 clips from each of `D_danya1`, `D_danya2`, `D_danya4`
in train, 7 from `D_danya3` in val, and 7 from `D_danya` in test. Within each recording,
clips are selected at evenly spaced positions in the sorted available start offsets, including
the first and last. Labels and source split assignments are preserved. This avoids giving D
more training weight simply because more excerpts of D were available; it does not guarantee
an unbiased model. Training reads the chosen dataset only and does not merge these folders.

## Guarantees

**Clips**
- `.npy` (NumPy array file) of dtype int16, shape **exactly (48 000,)**, i.e. 16 kHz mono PCM16
  samples of 3.000 s; loudness is the source's (no gain or loudness normalization).
- At least **60 % speech** per clip. Speech = 20 ms frames whose level exceeds max(−45 dBFS, p20 + 8 dB),
  with p20 the 20th percentile of frame levels over the whole source recording. Silence before the first
  and after the last speech frame of a recording is trimmed. The producer guarantees this; it can't be
  re-measured on a 3 s clip alone, because the threshold depends on the whole recording.

**Splits**
- Each `path` exists, is unique and appears in exactly one split.
- All clips cut from one source recording are in the same split (no clip-level leakage). Splits are by
  recording, not by session/day: different recordings of the same day may land in different splits.
- The builder targets about 70/15/15 of speech duration per speaker across train/val/test;
  val and test only get recordings that yield at least one clip, and at least one recording stays in
  train. A speaker with fewer than 3 usable recordings goes to train only. Recordings that yield no
  clip are dropped (`iml-data build` prints how many).
- The builder generates train windows at 1.5 s hops (50 % overlap) and val/test windows at
  3 s hops. Filtering or subsampling can leave larger gaps between retained clips.
- Classes are optional: a dataset, and each of its splits, may hold only `allowed = 1` or only
  `allowed = 0` (e.g. an enrollment dataset filtered to allowed speakers). `data/our_voices`
  currently contains both classes; enrollment must filter to `allowed=1`. Training
  requires both classes in train and val (see below).

## Producing and checking a dataset

The original raw recordings and source CSVs for the existing shared datasets are unavailable
in this checkout and mounted data directory. Telegram collector code exists in Git history
but is absent from this checkout. Building a new dataset requires separately supplied raw
recordings and labels; the stored clips cannot recover the original recordings in full.

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
recordings/N174_174-168635-0003.flac,N174,0
```

- `path` is relative to `sources.csv`: always `recordings/<file>` (POSIX, no `..`, no absolute paths).
- The file name (without extension) is the recording id, so it must be unique; clip names start with it.
- `allowed` is `1`/`0` and must be the same for all recordings of a speaker.

The pipeline lives in `src/iml/preprocess.py`:
1. decode to 16 kHz mono int16, the Telegram collector's recipe;
2. VAD and trimming;
3. per-speaker split by whole recordings;
4. 3 s windows with the speech threshold, saved as int16 `.npy` clips.

The builder writes clips and CSVs to a temporary sibling folder before moving them into the
output dataset; the CLI validates the result afterward. Raw recordings must not live inside
the output folder. Validation lives in `src/iml/contract.py`. The builder adds no background
noise, reverberation or pitch augmentation and does not normalize waveform loudness.

## Using a dataset

```bash
uv run iml-train --config app/models/configs/resnet18.json --data data/training_dataset_v2 --member <you> --topic <topic>
export IML_DATA=data/training_dataset_v2                                         # or once per shell
```

- `load_splits` validates the folder and builds one `SpectrogramDataset` per split. It refuses a
  dataset whose train or val split lacks a class (FAR/FRR/EER and the decision threshold are
  undefined without both) and warns if test does (its FAR/FRR/EER are then NaN).
- CSV paths resolve relative to the selected dataset folder. For example, `audio/file.npy`
  resolves to `data/training_dataset_v2/audio/file.npy` when that folder is selected. Relative
  `--data` paths resolve from the process working directory.
- `ExperimentConfig.excerpt_sec` defaults to 3.0, matching stored clips. Training computes
  log-mel spectrograms and configured spectrogram normalization in memory without saving
  modified audio. `cfg.augment` is currently unused by the loader.
- Each run writes the dataset fingerprint (manifests + clip bytes) to `data.json`. A finished run is
  only reused for the same dataset; for other data, or for runs from before data tracking (no
  `data.json`), `iml-train`/`iml-sweep` stop and ask for `--overwrite`.
- In notebooks: `load_splits(cfg, data_dir=DATA)`, and `check_run_data` / `record_run_data` from
  `iml.cli` for the same guard (see `notebooks/_template`).

## Changing the contract

Changes to the format or contract guarantees bump `CONTRACT_VERSION` in `src/iml/contract.py`
and the version in this file's title. Dataset contents or count updates alone do not require a
contract version bump, but require recounting and validation. The dataset fingerprint changes
with its manifests or clip bytes.

- v3: clips are `.npy` int16 arrays instead of WAV. Same samples, different bytes, so the fingerprint
  salt changed and runs recorded on v2 datasets are not reused.
- v2: single-class datasets and splits are valid (v1 required both classes in every split). Every v1
  dataset is a valid v2 dataset with the same fingerprint, so finished runs stay reusable.

## Known limitations

- MS was recorded on a different phone and codec (AAC `.m4a`) than the other allowed speakers
  (Telegram Opus voice notes), and the strangers (N) come from yet another source. A model can learn
  the **channel** instead of the voice; record negatives the same way as positives before trusting FAR.
