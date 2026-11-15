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

## Main dataset: training_main

**`data/training_main` is the main dataset for two-stage fine-tuning: VoxCeleb backbone
pretraining followed by binary allowed/invalid head tuning.** `data/training_dataset_v2`
is outdated as the main training dataset and remains a historical reference and the parent
of the inherited task clips. Select the main dataset with `--data data/training_main` or
`IML_DATA=data/training_main`. The CLI has no default dataset and never merges folders.
Changing the primary dataset does not change contract version 3.

Counts checked against the manifests and provenance on **2026-11-15**:

| Split | Allowed clips | Invalid clips | Total clips | Invalid / allowed |
|---|---:|---:|---:|---:|
| Train | 228 | 73,269 | 73,497 | 321.36 |
| Validation | 28 | 6,395 | 6,423 | 228.39 |
| Test | 28 | 12,750 | 12,778 | 455.36 |
| **Total** | **284** | **92,414** | **92,698** | |

Recalculate these counts and update this document and README whenever `training_main`
data, labels, or split membership changes. Counts are a snapshot, not fixed contract requirements.

### Recording sources and provenance

| Source | Source recording IDs | Speaker IDs | Train clips | Val clips | Test clips | Total clips |
|---|---:|---:|---:|---:|---:|---:|
| Inherited v2 task data | 259 | 45 | 513 | 63 | 63 | 639 |
| VoxCeleb1 | 13,730 | 789 | 26,289 | 2,279 | 4,553 | 33,121 |
| VoxCeleb2 | 26,391 | 1,425 | 46,695 | 4,081 | 8,162 | 58,938 |
| **Total** | **40,380** | **2,259** | **73,497** | **6,423** | **12,778** | **92,698** |

These are metadata identity counts, not proof of unique real-world people across sources.
Several clips can come from the same recording; clip counts do not measure independent
recordings or unique audio duration.

- **Inherited task data:** every v2 clip is preserved with its original waveform bytes,
  filename, label, and split. M, P, MS, and D supply the 284 allowed clips. T supplies 71
  invalid clips, and 40 external N speaker IDs supply another 284 invalid clips. Historical
  recording/channel details and per-speaker counts remain below.
- **VoxCeleb:** 40,121 retained originals from 2,214 speaker IDs, selected using
  `data/sv_clean/sources.parquet` (`keep=True`, `source` equal to `vox1` or `vox2`) and
  `speakers.parquet`. Original WAV/M4A members come from `ProgramComputer/voxceleb` on
  Hugging Face. This is the curated SV selection, not the whole repository. TidyVoice is
  excluded, and precomputed SV spectrograms are not used as the training inputs.
  All 92,059 added clips have `allowed=0`; invalid is an access label, not a quality label.

The external originals are decoded to 16 kHz mono int16 without loudness normalization.
Recording-level VAD retains three-second windows with at least 60% speech, using the
shared processing rules under Guarantees. Exact duplicate waveforms are excluded from
the added clips. All 40,121 selected originals produced at least one retained clip in
this build; no oversampling is stored in the dataset.

The two-column binary manifests remain the model input index. Additional files describe
the dataset without changing that format:

- `clip_provenance.csv`: `path`, `allowed`, `split`, `speaker`, `recording_id`, `origin`,
  `start_ms`, and `sha256_waveform`; external rows also include `origin_url`, `origin_member`,
  `sha256_original`, and `original_sv_split`. Origins are `training_dataset_v2`, `vox1`, `vox2`.
- `external_recordings.csv`: one row per selected external original, including source,
  speaker, assigned split, candidate/retained clip counts, and any dropped reason.
- `build_summary.json`: source fingerprints, seed, split targets, parent counts, and actual
  completed counts. `progress.sqlite` is a build journal, not a training input.

The standard binary loader ignores these sidecars. Stage-1 data selection must join the
manifest to `clip_provenance.csv` by `path` to obtain origin and speaker labels. The stored
`original_sv_split` is historical provenance; use the current manifest/`split` assignment.

### Current split and its limits

- New VoxCeleb speakers stay wholly in one split. Seed **42** targets **70/10/20 of original
  recording duration within each corpus**, rather than a fixed proportion of clips.
- External speaker counts are **1,550 train / 221 val / 443 test**: VoxCeleb1 contributes
  552/79/158 and VoxCeleb2 contributes 998/142/285. External recording counts are
  28,226/3,957/7,938, respectively.
- Training windows use **1.5 s hops** (50% overlap); validation/test use **3 s hops**.
  VAD, whole-speaker assignment, and these different hops make the final overall clip
  shares **79.29% / 6.93% / 13.78%**. The 639 inherited clips keep their approximately
  80/10/10 split without reassignment.
- The 2026-11-15 audit found no manifest/provenance label or split mismatches, no repeated
  manifest paths, no source recording IDs crossing splits, and no duplicate stored waveform
  hashes. Every waveform passed the contract validator. Hash checks used stored provenance;
  they do not establish absence of acoustic near duplicates or shared sessions.
- Speaker separation applies to the added VoxCeleb IDs, not every inherited speaker:
  **37 of 40 N IDs cross splits**. T is also present across splits. These examples measure
  rejection partly on familiar impostors. Allowed M/P/MS/D identities intentionally occur
  in all splits using different recordings, matching the fixed access list.
- **39 VoxCeleb source video IDs cross splits**, involving 112 original recording rows
  associated with different speaker IDs. Recording IDs are separate, but some background
  and session conditions can be shared. Different IDs across corpora also do not guarantee
  different real-world identities. Stronger evaluation would group shared videos together.
- Validation and test each contain only **28 allowed clips** from **6 and 8 recording IDs**,
  respectively. One false rejection changes clip-level FRR by about **3.57 percentage points**;
  clips from a recording are correlated. More negatives do not improve this positive coverage.

### Using the split for two-stage fine-tuning

The current train/val/test assignment is the **shared outer split for the whole pipeline**.
Only train recordings may update model parameters in either stage. Validation selects
settings/checkpoints and the final binary threshold; test is reserved for final evaluation.
This restriction also applies to self-supervised pretraining.

1. **Stage 1: pretrain the spectrogram backbone.** Select the VoxCeleb portion of train by
   `origin` (`vox1`/`vox2`). Use provenance speaker IDs as multiclass supervision, or an
   explicitly chosen self-supervised objective. Using only `allowed=0` as the pretraining
   target gives the model the trivial solution of always rejecting. Speaker-ID pretraining
   needs an inner validation split of held-out recordings/videos of the training identities;
   retain training examples for each such identity. The outer VoxCeleb validation/test
   speakers are unseen and are unsuitable for ordinary closed-set classification accuracy
   (embedding verification is a different evaluation). No pretraining script implements this
   stage yet; `load_splits` and the current binary CLI require both binary classes in train/val.
2. **Stage 2: tune the allowed/invalid head.** Transfer the pretrained backbone, replace the
   pretraining classifier with a binary head, freeze the backbone, and train the head using
   allowed and invalid examples from the same outer train split. A balanced selection has
   **228 allowed + 228 invalid clips**; source/speaker-aware selection should retain local T
   and inherited N negatives alongside any VoxCeleb negatives. Leave outer validation/test
   unchanged and report FAR/FRR/EER and macro-F1, including FAR by negative source.

[`src/iml/imbalance.py`](src/iml/imbalance.py) supplies `imbalance_report`, `balanced_indices`,
`balanced_subset`, `freeze_backbone`, and `keep_frozen_bn_eval` for stage-2 building blocks.
Its random balancing enforces class counts only, not source/speaker representation.
With the current train manifest order and default `cfg.seed=42`, it selects 228 VoxCeleb
negatives and no inherited negatives. Apply explicit negative selection for task coverage.

The `fit` function in [`src/iml/train.py`](src/iml/train.py) reads
`ExperimentConfig.finetune_from`: it loads the entire state dict from a compatible
binary checkpoint, freezes the backbone, and tunes the existing head on a balanced training
subset sampled once per run. Use `fit(cfg.with_overrides("finetune_from=<run>", "lr=1e-4"), loaders)`.
It does not yet load just the backbone from a multiclass speaker checkpoint or replace that
head. Those adaptations are still required for the intended two-stage pipeline; the existing
binary-only two-phase recipe is not stage-1 VoxCeleb pretraining. See the
[README workflow](README.md#two-stage-fine-tuning-with-training_main).

## Historical dataset: training_dataset_v2 (outdated)

V2 is retained for earlier experiments and provenance; use `training_main` for new work.
All counts and sampling details in this section describe the inherited v2 subset only.
The following counts were recalculated from its CSVs on 2026-11-01:

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
These historical counts describe v2 only; recount them if v2 itself changes. Validate any
dataset after changes.

`data/our_voices` is the broader team voice collection, including T negatives and 38 anonymous
clips whose identities remain unresolved. It is not the unified training source; enrollment
banks must use only verified allowed speakers with `allowed=1`. The `sv_clean`, `sv_nosil`
and `sv_noise` datasets use speaker-identity Parquet manifests and precomputed spectrogram
arrays, so they cannot be passed directly to the current binary training loader. The
`sv_clean` source/speaker metadata does supply provenance for the added VoxCeleb clips
in `training_main`; this does not make the SV dataset format a binary-loader input.

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
- These inherited prefixes also occur in `training_main`. Added external clips use
  `vox1_id<speaker>_<video>_<segment>_<start_ms>.npy` or the corresponding `vox2_` prefix;
  use `clip_provenance.csv` for the authoritative speaker, recording, origin, and split.

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
- All clips cut from one source recording are in the same split (no clip-level leakage).
  Grouping follows the dataset-specific policy above; separation by session/day is not
  guaranteed, and different recordings of the same day may land in different splits.
- The general `iml-data build` command targets about 70/15/15 of speech duration per speaker
  across train/val/test; this is not the `training_main` VoxCeleb assignment described above.
  Val and test only get recordings that yield at least one clip, and at least one recording stays in
  train. A speaker with fewer than 3 usable recordings goes to train only. Recordings that yield no
  clip are dropped (`iml-data build` prints how many).
- The builder generates train windows at 1.5 s hops (50 % overlap) and val/test windows at
  3 s hops. Filtering or subsampling can leave larger gaps between retained clips.
- Classes are optional: a dataset, and each of its splits, may hold only `allowed = 1` or only
  `allowed = 0` (e.g. an enrollment dataset filtered to allowed speakers). `data/our_voices`
  currently contains both classes; enrollment must filter to `allowed=1`. Training
  requires both classes in train and val (see below).

## Producing and checking a dataset

The original raw recordings and source CSVs for the historical v2 task data are unavailable
in this checkout and mounted data directory. Downloaded VoxCeleb originals are cached
separately from the processed `training_main` clips. Telegram collector code exists in Git history
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
uv run iml-train --config app/models/configs/resnet18.json --data data/training_main --member <you> --topic <topic>
export IML_DATA=data/training_main                                         # or once per shell
```

- `load_splits` validates the folder and builds one `SpectrogramDataset` per split. It refuses a
  dataset whose train or val split lacks a class (FAR/FRR/EER and the decision threshold are
  undefined without both) and warns if test does (its FAR/FRR/EER are then NaN).
- CSV paths resolve relative to the selected dataset folder. For example, `audio/file.npy`
  resolves to `data/training_main/audio/file.npy` when that folder is selected. Relative
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
