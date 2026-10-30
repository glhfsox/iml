# IML voice intercom (speaker verification on spectrograms)

Binary task from the course project description (`2026_2027_IML_project_descr.pdf`):
class 1 = allowed person, class 0 = not allowed, input = log-mel spectrogram of a short
recording, models = CNNs trained from scratch, metrics = FAR / FRR on train, val and test.

## Setup

```bash
uv sync --group dev          # macOS: PyPI torch (MPS); Linux server: HIP/ROCm torch for the AMD GPU (see pyproject)
uv run pytest                # fast CPU tests: config, metrics, data contract (no model building: models are checked by real GPU runs)
make smoke DATA=<dataset>    # two short runs + comparison table under runs/smoke/
```

On the AMD server check the ROCm version first (`rocm-smi --showversion`) and, if it is not
6.4, change the `pytorch-rocm` index URL in `pyproject.toml` (e.g. `.../whl/rocm6.3`).

GPU note: PyTorch drives the AMD card through HIP but names the device `cuda`
(`torch.device("hip")` does not exist). `get_device()` picks it automatically, `--device hip`
is accepted as an alias, `env.json` records `backend: hip` and the ROCm version, and AMP /
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
data/           raw recordings (gitignored); datasets follow DATA_CONTRACT.md
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
uv run iml-train --config app/models/configs/resnet18.json --member <you> --topic optimizers --set optimizer=sgd --set lr=0.01
# a grid
uv run iml-sweep --config app/models/configs/convnext_tiny.json --grid app/models/configs/grids/<arch>_lr.json --member <you> --topic lr
# compare everything under a folder -> table.md/csv + loss.png + val_metrics.png
uv run iml-compare runs/<you>/lr
```

In a notebook the same thing is three cells: build a list of configs, call `fit(cfg, loaders)`
for each, call `summary_table` / `plot_curves`. Copy `notebooks/_template/` to
`notebooks/<you>/` and `architecture/` to `<arch>/` (one notebook = one architecture, a grid of
parameter configs; see `notebooks/_template/architecture/experiments.ipynb`).

Every run writes `config.json`, `metrics.json` (FAR/FRR/EER/F1/accuracy per split),
`history.csv`, `loss.png`, `metrics.png`, `best.pt`, `last.pt`, `env.json`, `predictions_<split>.npz`
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

Data: every dataset follows [DATA_CONTRACT.md](DATA_CONTRACT.md) (`train/val/test.csv` with
`path,allowed` + 3 s clips stored as `.npy` int16 arrays of 48000 samples, contract v3). Build one with
`iml-data build`, check it with `iml-data validate`, and train on it with `--data <dataset>` (or
`$IML_DATA`); in notebooks `load_splits(cfg, data_dir=...)`.

`data/our_voices` is a collage of only our own recordings, with no external or LibriSpeech speakers. Allowed:
team members M, P, T, D and MS (renamed from S): our earlier voice notes plus the new P recordings from
`data/training_dataset_v2`, deduplicated per recording. The speaker banks for
zero-shot identification are built from it.

## Server workflow

```bash
make sync SERVER=<ssh host>      # rsync code + configs (no data, runs, venv)
ssh <server> 'cd iml && uv sync && tmux new -d -s train "uv run iml-train --config app/models/configs/resnet18.json --member me --topic x"'
make pull-runs SERVER=<ssh host> # bring runs/ back
```

## Production demo

```bash
uv run python -m app.main --run runs/<member>/<topic>/<run> --wav recording.wav  # PyTorch best.pt (dev)
make freeze RUN=runs/<member>/<topic>/<run>     # freeze: copy model.onnx into app/models/speaker_verification/<model>/
uv run python -m app.main --model resnet18 --wav recording.wav                 # frozen graph, onnxruntime
```

A frozen model folder holds only `model.onnx`: the run's config and decision threshold are stored
in the graph's metadata, and the app runs it with onnxruntime (log-mel front end in PyTorch on
CPU, no MC dropout). `--run` keeps the PyTorch path and also accepts a `.onnx` file.
