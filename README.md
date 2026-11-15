# IML voice intercom (speaker verification on spectrograms)

Binary task from the course project description (`2026_2027_IML_project_descr.pdf`):
class 1 = allowed person, class 0 = not allowed, input = log-mel spectrogram of a short
recording, models = CNNs trained from scratch or fine-tuned from ImageNet, metrics = FAR / FRR on train, val and test.

## Setup

```bash
uv sync --group dev
uv run pytest                # fast CPU tests: config, metrics, data contract (no model building: models are checked by real GPU runs)
make smoke DATA=data/training_main    # two short binary runs + comparison table under runs/smoke/
```

GPU note: PyTorch drives the AMD card through HIP but names the device `cuda`
(`torch.device("hip")` does not exist). `get_device()` picks it automatically, `--device hip`
is accepted as an alias, and AMP /
`torch_compile` work the same way as on NVIDIA. Verify on the server with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"
```

## Layout

```
src/iml/        shared library: config, 5 models + declarative custom, data, metrics, train loop, run folders, CLIs
app/            production side: load one best model and classify a recording (app/models/ = frozen weights)
notebooks/      notebooks/<member>/main.py + <arch>/experiments.ipynb (one architecture, a grid of parameter configs)
app/models/configs/  baseline json per model + grids/ for sweeps (gitignored: regenerate with `make configs` or share by hand)
runs/           <member>/<topic>/<run_name>/  — results; not in git (shared via `make pull-runs`)
data/           shared processed datasets (gitignored); training_main/ is the main dataset
tests/          pytest (config, metrics, data contract; nothing trains or builds a model here)
docs/architecture/map.jsonl   component map
```

## Models under comparison

| name               | source                       | params | why                                             |
|--------------------|------------------------------|--------|-------------------------------------------------|
| `resnet18`         | torchvision ResNet-18        | 11M    | baseline; `pretrained=true` = transfer learning |
| `efficientnetv2_s` | timm `efficientnetv2_rw_s`   | 22M    | compound-scaled MBConv / FusedMBConv            |
| `convnext_tiny`    | timm `convnext_tiny.fb_in1k` | 28M    | modernised ResNet, 7x7 depthwise convs          |
| `resnest26d`       | timm `resnest26d.gluon_in1k` | 15M    | split-attention ResNet                          |
| `regnet_y_800mf`   | timm `regnety_008_tv.tv2_in1k` | 5.6M | RegNet-Y-800MF (torchvision weights), the light one |
| `custom`           | `arch` list of `models/blocks.py` blocks | —  | your own CNN, declared in the config (`pretrained=false`) |

All take `(B, 1, 80, T)` log-mels (`n_mels=80`) and return 2 logits; any `T` (excerpt length) works.

Custom models: `src/iml/models/blocks.py` is a block library (one module) for assembling your own
architectures, either in Python or declaratively: `model="custom"` plus `arch=[{"type": "conv_stem", "out_ch": 32}, ...]`
in the config (channels are inferred, the list ends with a pooling block, the classifier head is added).
See [the models README](src/iml/models/README.md); `make configs` writes an example `custom.json`.

## Running experiments

One config = one `ExperimentConfig` (see `src/iml/config.py`). The run folder name is
`<model>_<non-default fields>_<hash>`, so two identical configs map to the same folder and are
not retrained.

```bash
# one run, headless (use tmux on the server)
uv run iml-train --config app/models/configs/resnet18.json --data data/training_main --member <you> --topic optimizers --set optimizer=sgd --set lr=0.01
# a grid
uv run iml-sweep --config app/models/configs/convnext_tiny.json --grid app/models/configs/grids/<arch>_lr.json --data data/training_main --member <you> --topic lr
# compare everything under a folder -> table.md/csv + loss.png + val_metrics.png
uv run iml-compare runs/<you>/lr
```

In a notebook the same thing is three cells: build a list of configs, call `fit(cfg, loaders)`
for each, call `summary_table` / `plot_curves`. Copy `notebooks/_template/` to
`notebooks/<you>/` and `architecture/` to `<arch>/` (one notebook = one architecture, a grid of
parameter configs; see `notebooks/_template/architecture/experiments.ipynb`).

Every run writes `config.json`, `metrics.json` (FAR/FRR/EER/F1/accuracy per split),
`history.csv`, `loss.png`, `metrics.png`, `best.pt`, `last.pt`, `predictions_<split>.npz`
and `model.onnx` (exported from `best.pt` at the end of `fit` and verified against PyTorch with
onnxruntime; `export_onnx=false` skips it, `make export RUN=...` redoes it by hand).
Set `mc_dropout_samples` > 0 (with `dropout` > 0) to evaluate with Monte-Carlo dropout.

Evaluation protocol: the checkpoint with the lowest validation EER is kept; the decision
threshold is the FAR == FRR point on the validation set (`decision_threshold` in
`metrics.json` and in `best.pt`) and is applied unchanged to train/val/test and by `app/`.
`class_weight=balanced` reweights the cross-entropy by inverse class frequency. A finished
run folder is loaded instead of retrained (`--overwrite` / `fit(..., overwrite=True)` to redo).
With `pretrained=true` (the default) the ImageNet weights are loaded and `init` is ignored, so
`build_model` refuses `pretrained=true` with a non-default `init`: a sweep cannot produce fake
comparisons. Set `pretrained=false` to train from scratch with the chosen `init`.

Include a class-imbalance report with each binary experiment:
`imbalance_report(loaders["train"])["ratio"]` gives the majority/minority ratio.
For imbalance-aware binary training, use `loss=focal|soft_f1` or `class_weight=balanced`.
Under strong imbalance compare runs by EER/FAR/FRR and macro-F1 (on the same test set), not by
accuracy or loss values (CE/focal/soft-F1 are training objectives with different scales).

The commands above run the existing binary trainer. They do not implement the VoxCeleb
pretraining stage described below.

## Main dataset: training_main

**`data/training_main` is the main dataset for the planned two-stage fine-tuning workflow.**
`data/training_dataset_v2` is outdated as the main training dataset; retain it for historical
experiments and as the preserved parent of `training_main`. Select the main dataset with
`--data data/training_main` or `IML_DATA=data/training_main`; notebooks use
`load_splits(cfg, data_dir="data/training_main")` for binary training.
The loader reads only the selected folder and never merges datasets automatically.
There is no implicit CLI default.

Counts checked against the manifests and provenance on **2026-11-15**:

| Split | Allowed clips | Invalid clips | Total clips | Invalid / allowed |
|---|---:|---:|---:|---:|
| Train | 228 | 73,269 | 73,497 | 321.36 |
| Validation | 28 | 6,395 | 6,423 | 228.39 |
| Test | 28 | 12,750 | 12,778 | 455.36 |
| **Total** | **284** | **92,414** | **92,698** | |

The dataset combines **639 inherited v2 clips** with **92,059 new invalid-person clips**
from **40,121 VoxCeleb originals**. The new sources are VoxCeleb1 (13,730 recordings,
789 speaker IDs) and VoxCeleb2 (26,391 recordings, 1,425 speaker IDs), selected from retained
entries in `data/sv_clean/sources.parquet`. Their original audio comes from
`ProgramComputer/voxceleb`; the dataset uses the curated selection rather than the entire
repository or the SV spectrogram arrays. TidyVoice is excluded. All inherited clips retain their
waveforms, labels, filenames, and splits. M, P, MS, and D are allowed; T, N speakers, and
VoxCeleb speakers are invalid. Here, "invalid" means no access, not poor audio quality.

New VoxCeleb speakers are assigned wholly to one split, targeting **70/10/20 by original
recording duration within each corpus**, with seed 42. Their train/validation/test speaker
counts are **1,550 / 221 / 443**. Training uses 1.5-second clip hops; validation/test use
3-second hops. Consequently, overall clip shares are **79.29% / 6.93% / 13.78%**, rather
than 70/10/20. The inherited v2 subset retains its approximately 80/10/10 clip split.

The current split is usable as the shared outer split for both stages: source recording IDs
do not cross splits, and stored waveform hashes contain no exact duplicates. It is not fully
separate by session or source video: **39 VoxCeleb video IDs cross splits**, and **37 of the
40 inherited N speaker IDs occur in multiple splits**. Allowed identities intentionally occur
in every split with different recordings, matching the fixed access list. Each validation/test
split has only 28 allowed clips, so one false rejection changes clip-level FRR by about
3.57 percentage points; excerpts from one recording are correlated.

Waveform datasets follow [DATA_CONTRACT.md](DATA_CONTRACT.md): `train/val/test.csv` with
`path,allowed` plus `.npy` int16 arrays of 48,000 samples. CSV paths resolve relative to the
dataset folder. `clip_provenance.csv` adds speaker IDs, source recording IDs, origins, offsets,
and hashes; `external_recordings.csv` records processing outcomes for external originals;
`build_summary.json` records build settings and counts. The binary loader ignores these sidecars.
Validate the main training source with:

```bash
uv run iml-data validate data/training_main
```

All 92,698 clips passed the dataset validator in the 2026-11-15 audit. See
[DATA_CONTRACT.md](DATA_CONTRACT.md) for source/split tables and the historical v2 details.
Recalculate the counts and update both documents whenever the data, labels, or splits change.

## Two-stage fine-tuning with training_main

1. **Pretrain the CNN backbone on VoxCeleb speech.** Select only VoxCeleb entries from
   `train.csv`, joining `clip_provenance.csv` by `path` and filtering `origin` to `vox1`/`vox2`.
   Use the provenance `speaker` identity as a multiclass target, or an explicitly chosen
   self-supervised objective, to learn useful spectrogram/voice features. The stored
   `allowed=0` labels remain correct for the final access task, but training only on this one
   binary label can be solved by always rejecting. For speaker-ID classification, derive an
   inner validation split from held-out recordings/videos of the training identities; the
   outer validation/test VoxCeleb identities are unseen and cannot be evaluated with ordinary
   closed-set speaker-classification accuracy. No stage-1 pretraining script currently exists.
2. **Tune the binary access head.** Transfer the learned backbone, replace the pretraining
   head with an allowed/invalid head, freeze the backbone, and train the head on allowed and
   invalid examples drawn only from `train.csv`. A balanced subset has 228 allowed and 228
   invalid clips. Select negatives across speakers and sources so T and inherited N examples
   are represented; uniform random sampling from the huge invalid pool can miss them.
   Keep the outer validation/test membership unchanged, choose the binary threshold on
   validation, and evaluate the final model on test. Report FAR by negative source alongside
   overall FAR/FRR/EER and macro-F1.

Validation/test recordings must remain excluded from training in **both stages**, including
self-supervised pretraining. The current outer split can be reused; each stage selects its
training subset without randomly repartitioning the whole dataset.

Existing code provides parts of stage 2:

- [`src/iml/data.py`](src/iml/data.py): waveform-to-log-mel frontend and dataset/loader helpers.
- [`src/iml/imbalance.py`](src/iml/imbalance.py): `imbalance_report`, `balanced_indices`,
  `balanced_subset`, `freeze_backbone`, and `keep_frozen_bn_eval`. Balancing is by class only;
  source/speaker quotas require an explicit selection before or instead of random balancing.
- [`src/iml/train.py`](src/iml/train.py): `fit` with `finetune_from` loads a compatible binary
  checkpoint, freezes the backbone, and tunes the existing head on a balanced train subset
  drawn once from `cfg.seed`. It loads the full state dict and does not replace a multiclass
  speaker head or transfer a stage-1 backbone automatically. That checkpoint/head adaptation
  still needs implementation. With the current manifests and default seed 42, automatic
  balancing picks 228 VoxCeleb negatives and no inherited negatives.

The earlier binary-only two-phase experiment remains available: run `rec = fit(cfg, loaders)`
from ImageNet (`pretrained=true`, `loss=ce`, `class_weight=none`), then
`fit(cfg.with_overrides(f"finetune_from={rec.path}", "lr=1e-4"), loaders)` with the same model
and plain CE. It first trains the whole binary network, then tunes its head. This existing
recipe does not perform the VoxCeleb-only representation pretraining planned above.

## Historical datasets

The outdated `training_dataset_v2` contains **639 clips from 259 source recording IDs**:
284 allowed clips (71 each for M, P, MS, D) and 355 invalid clips (71 T, 284 from 40 N
speakers). Its train/validation/test totals are 513/63/63. These same clips are preserved
in `training_main`. MS was formerly named MH; waveform comparisons confirmed its audio
against `our_voices`. D's 71 clips span all five D recordings. See the data contract for
the original per-speaker counts and sampling history.

`data/our_voices` retains the broader team voice collection, including T negatives and anonymous
clips with unresolved identities. It is not the unified training source; enrollment banks must
filter to verified allowed speakers. `data/sv_*` uses speaker-identity Parquet manifests and
precomputed spectrograms and cannot be loaded directly by `iml-train`.

Original raw audio and source CSVs for v2 are unavailable in this workspace
and mounted data directory. Downloaded VoxCeleb originals are cached separately from the
processed clips. `iml-data build` can build datasets from separately supplied raw recordings
and labels; Telegram collector code exists only in Git history. The stored dataset has no
added noise/reverb/pitch augmentation, and `cfg.augment` is unused by the current training loader.

## Exporting spectrogram pictures

[export_spectrograms.py](src/iml/export_spectrograms.py) saves one PNG per clip using the
same log-mel transform and normalization as the training loader. Select any dataset that
follows [DATA_CONTRACT.md](DATA_CONTRACT.md):

```bash
# Default output: data/training_main/spectrograms/
uv run python -m iml.export_spectrograms --data data/training_main

# Choose a separate output folder
uv run python -m iml.export_spectrograms --data data/training_main --out /tmp/training-main-spectrograms

# Use an experiment's frontend settings and choose picture resolution
uv run python -m iml.export_spectrograms \
  --data data/training_main --config path/to/config.json --dpi 150
```

Images are grouped by split and allowed/invalid class, preserving each clip's relative path.
Only PNGs are saved; no CSV index or JSON metadata is generated.
For example, an allowed training clip at `audio/clip.npy` becomes
`<out>/train/allowed/audio/clip.png`.
Defaults use 80 mel bands, FFT 512, hop 160, and per-sample normalization. Pass
`--config path/to/config.json` to use an experiment's frontend settings. Pictures share fixed
color limits within each normalization mode, so their colors can be compared across clips;
values outside the display range saturate without changing model inputs.

Each run redraws pictures, replacing existing PNGs at the same paths so changed inputs and
settings take effect. `--dpi` controls image resolution and `--threads` limits CPU threads
(default: 4). Run with `--help` for all options. Pictures use deterministic evaluation
cropping; native three-second clips need no cropping.
The script preserves input waveforms, labels, and splits. The PNGs are for inspection:
models receive the numerical spectrogram tensors directly, rather than loading these images.

For dataset composition, audio-quality plots, and examples, see the
[recording EDA notebook guide](notebooks/spdsslg/eda/DATA_ANALYSIS.md).

## Server workflow

```bash
make sync SERVER=<ssh host>      # rsync code + configs (no data, runs, venv)
ssh <server> 'cd iml && uv sync && tmux new -d -s train "uv run iml-train --config app/models/configs/resnet18.json --data data/training_main --member me --topic x"'
make pull-runs SERVER=<ssh host> # bring runs/ back
```

## Model weights and the production demo

Weights are not in git (`*.pt`, `*.onnx` are ignored). After a clone, build the starting graphs
from the public ImageNet weights, then fine-tune and freeze your own:

```bash
make init-models                                # 5 x app/models/speaker_verification/<model>/model.onnx from ImageNet
                                                # weights (downloads once, deterministic; existing graphs are kept, FORCE=1 overwrites)
make init-model MODEL=convnext_tiny             # the same for one model only (you work on one architecture)
uv run iml-train --config app/models/configs/resnet18.json --data data/training_main --member <you> --topic <arch>
make freeze RUN=runs/<member>/<arch>/<run>      # replace the folder's model.onnx with the fine-tuned run's graph
make export RUN=runs/<member>/<arch>/<run>      # (optional) re-export a run's model.onnx by hand; fit already does it
uv run python -m app.main --model resnet18 --wav recording.wav                 # frozen graph, onnxruntime
uv run python -m app.main --run runs/<member>/<arch>/<run> --wav recording.wav  # PyTorch best.pt (dev)
```

All model targets live in `app/models/Makefile` (included by the root one): `configs`, `init-models`,
`init-model`, `export`, `export-all`, `freeze`. Fine-tuned graphs are shared outside git (rsync to the server, or a
release/LFS scheme if we add one); `make init-models` is always enough to run the app from scratch.

A frozen model folder holds only `model.onnx`: the run's config and decision threshold are stored
in the graph's metadata, and the app runs it with onnxruntime (log-mel front end in PyTorch on
CPU, no MC dropout). `--run` keeps the PyTorch path and also accepts a `.onnx` file.


