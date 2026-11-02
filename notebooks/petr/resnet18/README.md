# ResNet-18 training experiments and results

Branch: `feature/resnet18_training`. Base: `main`, commit `50cf7a5`.
The model has 11.17 million parameters. The experiment includes one smoke run and five main training runs.

## Work completed

1. Created a separate branch from main. Existing ConvNeXt experiment files remain separate.
2. Validated the dataset and saved a local snapshot and its fingerprint in `dataset_snapshot.json`.
3. Compared three learning rates with all other settings fixed.
4. Selected a configuration using validation metrics and repeated it with seeds 43 and 44.
5. Saved `best.pt`, exported `model.onnx`, and verified ONNX predictions against PyTorch.
6. Saved configs, metrics, epoch histories, dataset and environment metadata, and plots for all five main runs under `results/`.

## Dataset and comparison limits

The source is `data/training_dataset_v2`: 513 training clips (228 allowed, 285 not allowed), 63 validation clips, and 63 test clips (28 allowed and 35 not allowed in each).

The shared dataset changed after the earlier ConvNeXt experiments, which used 456/56/56 clips. Those results are not directly comparable. All five ResNet-18 runs have the same dataset fingerprint, matching the saved snapshot.

Clips are 3-second mono waveforms at 16 kHz, stored as int16 NumPy arrays. Audio is not included in Git. Training metrics may report `n=512` because the current training loader drops the incomplete final batch.

## Training configuration

- `pretrained=true`: fine-tuning ImageNet weights, rather than training from scratch. Check that this mode meets the course submission requirements.
- Optimizer: AdamW; batch size: 16; dropout: 0.2; weight decay: 0.05.
- Maximum 12 epochs; early stopping after four epochs without validation EER improvement.
- Learning-rate trials: 0.00005, 0.0002, and 0.0005, using seed 42.
- These settings match the earlier ConvNeXt series apart from the architecture.
- The selected learning rate was also evaluated with seeds 43 and 44.

## Learning-rate search

- `lr=0.00005`: validation EER 3.21%, loss 0.1413; best epoch 2; 6 epochs completed.
- `lr=0.0002`: validation EER 0.00%, loss 0.0560; best epoch 2; 6 epochs completed.
- `lr=0.0005`: validation EER 3.21%, loss 0.3760; best epoch 1; 5 epochs completed.

Selected `lr=0.0002`, seed 42, using the lowest validation EER, with validation loss as the tie-breaker. Additional seeds assess variability; neither test scores nor the best random seed were used for model selection.

Across seeds 42, 43, and 44, mean validation EER was 1.07%, with a sample standard deviation of 1.86 percentage points.

## Selected model: test results

The decision threshold was chosen only on validation: **0.730870**.

- Accuracy: **85.71%** (54/63 correct).
- FAR: **17.14%** (6/35 stranger clips accepted).
- FRR: **10.71%** (3/28 allowed clips rejected).
- Test EER: **17.50%**. EER uses its own threshold; the deployed decision threshold was not adjusted on test.

Validation results are substantially better than test results. The small dataset and validation/test gap limit confidence. More independent recordings are needed; recording channels and source differences may provide shortcuts unrelated to speaker identity.

## Interpreting settings and metrics

Learning rate controls the weight-update step size; it is the only hyperparameter varied in the initial search. Dropout and weight decay can reduce overfitting but were fixed in this series.

Batch size is the number of clips per batch. An epoch is a pass through the training data; a run trains a fresh model with one configuration. The optimizer updates weights, while hyperparameters are selected using validation data.

Lower EER, FAR, FRR, and loss are better; higher accuracy and F1 are better. During training, FAR/FRR are logged at threshold 0.5. Final metrics use the threshold selected on validation; see `results/*/metrics.json`. The comparison table sorts by F1 rather than EER.

## Reproduce

From the repository root:

```bash
uv sync --group dev
uv run python notebooks/petr/resnet18/run.py
```

On the original workspace, the runner automatically uses the saved local dataset snapshot. On another machine, supply a dataset with the recorded fingerprint:

```bash
uv run python notebooks/petr/resnet18/run.py --data /path/to/dataset
```

The runner checks the fingerprint, reuses completed configurations, and exports the selected model. A different dataset requires a separate experiment topic rather than overwriting these results.

Logs: `runs/petr/resnet18_experiments.log` and `runs/petr/resnet18_reproduce.log`.

## Model artifacts

`selected_run.txt` records the selected run directory, containing `best.pt` and `model.onnx`. Weights and audio are excluded from Git by repository rules. This branch publishes code, configs, metrics, and plots. Model weights remain in the Coder workspace.

## Suggested next experiment

Fix `lr=0.0002` and compare dropout values 0.0, 0.2, and 0.4 using the same validation split. Select settings on validation and use test for the final report. A fair architecture comparison requires both models to use the same dataset version.
