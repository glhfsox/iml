# ResNet-18 — Fine-tuning experiment report

## Objective and setup

Evaluate ResNet-18 as a baseline for the project's allowed / not-allowed voice classifier.
ImageNet-pretrained weights are fine-tuned on 3-second log-mel spectrograms
(16 kHz, 80 mel bins, FFT 512, hop 160, per-sample normalization).

- Dataset: `data/training_dataset_v2`, with 513 training, 63 validation and 63 test clips.
  Validation and test each contain 28 allowed and 35 not-allowed clips.
- Optimizer: AdamW; batch size 16; dropout 0.2; weight decay 0.05; no scheduler.
- Budget: up to 12 epochs, stopping after 4 epochs without improved validation EER.
- Hardware: AMD Radeon 8060S through PyTorch's ROCm backend.

## Experiments and results

Three seed-42 runs compare learning rates 0.00005, 0.0002 and 0.0005.
Their best-checkpoint validation EERs are **3.21%, 0% and 3.21%**, respectively.
Selection uses validation EER, then validation loss; test metrics do not select the configuration.

The selected learning rate is **0.0002**, seed 42, checkpoint at epoch 2.
Two further runs at that learning rate, seeds 43 and 44, give validation EERs of
**0% and 3.21%**. The five runs stop after 5–6 epochs. A separate earlier
1-epoch smoke test checked the pipeline and is excluded from this comparison.

At the selected checkpoint, the validation-derived decision threshold is **0.73087**.
Applied unchanged to test, it gives:

- Accuracy: **85.71%** (54/63 correct).
- FAR: **17.14%** (6/35 not-allowed clips accepted).
- FRR: **10.71%** (3/28 allowed clips rejected).

![Training and validation loss for the selected run](plots/loss.png)

![Validation EER across learning rates and seed repeats](plots/validation_eer.png)

## Interpretation

ResNet-18 is a workable baseline, but perfect validation EER does not imply reliable
performance: the holdouts are small, and test errors remain material. Variation across
seeds and the loss curves warrant further checks for overfitting and recording-level
separation. These runs do not establish deployment readiness.

The earlier ConvNeXt report used a different dataset (456/56/56 clips) and trained
from scratch. Its numbers are not a controlled architecture comparison with these runs.
A fair follow-up should use the same data and a matched training protocol.

## Reproduce

From the repository root, with the shared dataset mounted:

```bash
uv sync --group dev
uv run iml-data validate data/training_dataset_v2
uv run python analyze_resnet18.py --train
```

The script trains missing runs using the shared training loop, reuses matching finished
runs, selects the LR on validation only, and generates the two figures under
`runs/petr/resnet18_lr/report/`. Omit `--train` to regenerate figures from existing runs.
To publish updated figures, copy only those two PNGs into `reports/resnet18/plots/`.

All five recorded runs use dataset SHA1 `10ac1fb033e3b1845bc52ea2e2229c2982ad11f0`.
The script refuses different data; use `--data /path/to/original/snapshot` if the shared
dataset has changed. Training was performed at code commit `50cf7a5`; exact floating-point
results may vary across environments. Run configs, histories, checkpoints, predictions
and environment details remain in `runs/petr/resnet18_lr/` on Coder, outside Git.
