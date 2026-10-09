# IML voice intercom (speaker verification on spectrograms)

Binary task from the course project description (`2026_2027_IML_project_descr.pdf`):
class 1 = allowed person, class 0 = not allowed, input = log-mel spectrogram of a short
recording, models = CNNs trained from scratch, metrics = FAR / FRR on train, val and test.

## Setup

```bash
uv sync --group dev          # macOS: PyPI torch (MPS); Linux server: HIP/ROCm torch for the AMD GPU (see pyproject)
uv run pytest                # ~10 s, includes an end-to-end smoke training on synthetic data
make smoke                   # two dummy runs + comparison table under runs/smoke/
```

Add `--group data` to `uv sync` on a computer that will collect Telegram audio.  The training
server does not need Telegram credentials or a session file.

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
src/iml/        shared library: config, 5 models, data, metrics, train loop, run folders, MLflow, CLIs
app/            production side: load one best model and classify a recording (app/models/ = frozen weights)
notebooks/      one folder per member: main.py (demo) + comparisons/<topic>/experiments.ipynb
configs/        baseline json per model + grids/ for sweeps (gitignored: regenerate with `make configs` or share by hand)
runs/           <member>/<topic>/<run_name>/  — results; not in git (shared via MLflow and `make pull-runs`)
data/           reviewed recordings + prepared manifests (always gitignored)
tests/          pytest
docs/architecture/map.jsonl   component map
```

## Models under comparison

| name           | what                                                     | PDF points            |
|----------------|----------------------------------------------------------|-----------------------|
| `simple_cnn`   | CIFAR-tutorial-style baseline, every knob is a config field | b, d, f, g, h       |
| `cnn10`        | PANNs CNN10 (audio VGG), `depth` = number of blocks        | b, d                  |
| `resnet18`     | torchvision ResNet-18, 1-channel; `pretrained=true` = transfer learning | e, i     |
| `mobilenet_v2` | torchvision MobileNetV2, light enough for a laptop        | e                     |
| `ecapa_tdnn`   | ECAPA-TDNN speaker-verification net (1-D convs, SE, attentive pooling) | e, j     |

All take `(B, 1, n_mels, T)` and return 2 logits; any excerpt length works.

## Running experiments

One config = one `ExperimentConfig` (see `src/iml/config.py`). The run folder name is
`<model>_<non-default fields>_<hash>`, so two identical configs map to the same folder and are
not retrained.

```bash
# one run, headless (use tmux on the server)
uv run iml-train --config configs/cnn10.json --member <you> --topic optimizers --set optimizer=sgd --set lr=0.01
# a grid
uv run iml-sweep --config configs/simple_cnn.json --grid configs/grids/bn_position.json --member <you> --topic bn
# compare everything under a folder -> table.md/csv + loss.png + val_metrics.png
uv run iml-compare runs/<you>/bn
```

In a notebook the same thing is three cells: build a list of configs, call `fit(cfg, loaders)`
for each, call `summary_table` / `plot_curves`. Copy `notebooks/_template/` to
`notebooks/<you>/` and `comparisons/_topic/` to `comparisons/<topic>/`.

Every run writes `config.json`, `metrics.json` (FAR/FRR/EER/F1/accuracy per split),
`history.csv`, `loss.png`, `metrics.png`, `best.pt`, `last.pt`, `env.json`, `predictions_<split>.npz`.
Set `mc_dropout_samples` > 0 (with `dropout` > 0) to evaluate with Monte-Carlo dropout.

Evaluation protocol: the checkpoint with the lowest validation EER is kept; the decision
threshold is the FAR == FRR point on the validation set (`decision_threshold` in
`metrics.json` and in `best.pt`) and is applied unchanged to train/val/test and by `app/`.
`class_weight=balanced` reweights the cross-entropy by inverse class frequency. A finished
run folder is loaded instead of retrained (`--overwrite` / `fit(..., overwrite=True)` to redo).
Knobs a model does not use (e.g. `bn_position` for resnet18) must stay at their defaults;
`build_model` refuses such configs so a sweep cannot produce fake comparisons.

Until recordings exist, `--dummy` (CLI) / `DUMMY = True` (notebooks) use a synthetic dataset.

## Telegram data pipeline

The repository includes the local `voice-dataset` collector, but credentials, SQLite state,
review sheets and recordings stay under gitignored `data/`.  Install the optional collector and
check local FFmpeg:

```bash
uv sync --group dev --group data
uv run voice-dataset doctor
```

First rank personal chats without downloading media, then run the required 20-message pilot.
For four team members, three one-to-one chats normally cover the owner plus three teammates:

```bash
uv run voice-dataset rank-chats --start 2025-10-01 --end 2026-10-07
uv run voice-dataset collect-telegram \
  --start 2025-10-01 --end 2026-10-07 \
  --expected-chat-count 3 --limit 20
```

The command collects ordinary voice messages by default; video notes are included only with
`--include-video-notes`.  It stores originals, creates mono 16 kHz PCM16 WAV files, calculates
quality indicators, detects exact decoded-audio duplicates and resumes an interrupted fixed
batch. Automatic checks only route audio to accept/review/reject. The review sheet pre-fills
unconfirmed sender/day hints to make review faster, but a human still confirms the speaker,
consent and session in `data/labels/speaker_labels_<run>.csv`, then applies it:

```bash
uv run voice-dataset apply-labels data/labels/speaker_labels_<run>.csv
```

After review, build a one-vs-rest dataset for one owner:

```bash
uv run iml-prepare-data \
  --metadata /private/collector/data/metadata.csv \
  --source-root /private/collector \
  --target-speaker speaker_01 \
  --output data/processed/speaker_01
```

The command accepts only consented, manually approved, single-speaker, unique recordings and
validates mono 16 kHz PCM16 WAV files.  It splits every participant by whole recording sessions,
so all participants contribute to train/validation/test but no recording session can cross them.
Each person therefore needs at least three independent session codes (for example three days).

Train on the generated manifest:

```bash
uv run iml-train \
  --config configs/cnn10.json \
  --data-manifest data/processed/speaker_01/splits.json \
  --member <you> \
  --topic baseline
```

The dataset fingerprint becomes part of the experiment config/run hash, and the exact
`splits.json` is copied into the run folder.  See `docs/data_pipeline.md` for the schema,
collection boundary and privacy rules.

## MLflow (server)

```bash
make mlflow-ui                                   # on the server: sqlite backend, port 5000
export MLFLOW_TRACKING_URI=http://localhost:5000 # on the server, before iml-train / notebooks
ssh -L 5000:localhost:5000 <server>              # locally, then open http://localhost:5000
```

Without `MLFLOW_TRACKING_URI` nothing is tracked; the run folder is always the source of truth.

## Server workflow

```bash
make sync SERVER=<ssh host>      # rsync code + configs (no data, runs, venv)
ssh <server> 'cd iml && uv sync && tmux new -d -s train "uv run iml-train --config configs/cnn10.json --member me --topic x"'
make pull-runs SERVER=<ssh host> # bring runs/ back
```

## Production demo

```bash
uv run python -m app.main --run runs/<member>/<topic>/<run> --wav recording.wav
cp runs/<...>/best.pt runs/<...>/config.json app/models/   # freeze the final model
```
