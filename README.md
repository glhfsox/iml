# IML voice intercom (speaker verification on spectrograms)

Binary task from the course project description (`2026_2027_IML_project_descr.pdf`):
class 1 = allowed person, class 0 = not allowed, input = log-mel spectrogram of a short
recording, models = CNNs trained from scratch or fine-tuned from ImageNet, metrics = FAR / FRR on train, val and test.

## Setup

```bash
uv sync --group dev
uv run pytest                # fast CPU tests: config, metrics, data contract (no model building: models are checked by real GPU runs)
make smoke DATA=data/training_dataset_v2    # two short runs + comparison table under runs/smoke/
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
data/           shared processed datasets (gitignored); binary training uses training_dataset_v2/
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
uv run iml-train --config app/models/configs/resnet18.json --data data/training_dataset_v2 --member <you> --topic optimizers --set optimizer=sgd --set lr=0.01
# a grid
uv run iml-sweep --config app/models/configs/convnext_tiny.json --grid app/models/configs/grids/<arch>_lr.json --data data/training_dataset_v2 --member <you> --topic lr
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

Please add the short class-imbalance report with each experiment. Simply the ratio of instances of both would be enough.
Get it with `imbalance_report(loaders["train"])["ratio"]`; for imbalance-aware training set
`loss=focal|soft_f1` or `class_weight=balanced`, and for two-phase fine-tuning run `rec = fit(cfg, loaders)`
then `fit(cfg.with_overrides(f"finetune_from={rec.path}"), loaders)` (head only, balanced train subset).


`data/training_dataset_v2` is currently the **single source of unified data for binary model
training**. Select it with `--data data/training_dataset_v2` or set
`IML_DATA=data/training_dataset_v2`; notebooks use `load_splits(cfg, data_dir=...)`.
The loader reads only the selected folder and never merges datasets automatically.

As of 2026-11-01 it contains **639 preprocessed 3-second clips** from **259 source recording
IDs**. Valid speakers M, P, MS and D each contribute **71 clips** (57 train, 7 val, 7 test),
for 284 valid clips. Invalid clips total 355: 71 T clips and 284 clips from 40 external N
speakers. Overall splits contain 513 train, 63 val and 63 test clips. These are clip counts,
not counts of independent original recordings. **Recalculate and update these numbers when
new data arrives or data, labels, splits, naming, or the contract changes.**

Waveform datasets follow [DATA_CONTRACT.md](DATA_CONTRACT.md): `train/val/test.csv` with
`path,allowed` plus `.npy` int16 arrays of 48,000 samples. CSV paths resolve relative to the
dataset folder. Validate the current training source with:

```bash
uv run iml-data validate data/training_dataset_v2
```

MS was formerly named MH in `training_dataset_v2`; waveform comparisons confirmed the same
source audio as MS in `our_voices`. D's 71 clips span all five D source recordings, preserving
labels and splits. T has `allowed=0`. See the contract for the full count table and sampling.

`data/our_voices` retains the broader team voice collection, including T negatives and anonymous
clips with unresolved identities. It is not the unified training source; enrollment banks must
filter to verified allowed speakers. `data/sv_*` uses speaker-identity Parquet manifests and
precomputed spectrograms and cannot be loaded directly by `iml-train`.

Original raw audio and source CSVs for the existing datasets are unavailable in this workspace
and mounted data directory. `iml-data build` can build datasets from separately supplied raw
recordings and labels; Telegram collector code exists only in Git history. The builder adds
no noise/reverb/pitch augmentation, and `cfg.augment` is unused by the current training loader.

## Server workflow

```bash
make sync SERVER=<ssh host>      # rsync code + configs (no data, runs, venv)
ssh <server> 'cd iml && uv sync && tmux new -d -s train "uv run iml-train --config app/models/configs/resnet18.json --data data/training_dataset_v2 --member me --topic x"'
make pull-runs SERVER=<ssh host> # bring runs/ back
```

## Model weights and the production demo

Weights are not in git (`*.pt`, `*.onnx` are ignored). After a clone, build the starting graphs
from the public ImageNet weights, then fine-tune and freeze your own:

```bash
make init-models                                # 5 x app/models/speaker_verification/<model>/model.onnx from ImageNet
                                                # weights (downloads once, deterministic; existing graphs are kept, FORCE=1 overwrites)
make init-model MODEL=convnext_tiny             # the same for one model only (you work on one architecture)
uv run iml-train --config app/models/configs/resnet18.json --data data/training_dataset_v2 --member <you> --topic <arch>
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


