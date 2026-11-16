# ConvNeXt Tiny — Initial Experiment Report

**Project:** IML voice authorization classifier  
**Working branch:** `feature/testingConvNext`  
**Status:** Exploratory experiments; not a production-readiness assessment

## Objective

Evaluate whether ConvNeXt Tiny can be trained on the project's voice dataset, inspect generalization, and compare a small set of training settings (epochs, dropout, learning rate, and random seed).

## Setup

- **Model:** ConvNeXt Tiny (~27.82 million parameters)
- **Hardware:** AMD Radeon 8060S, accessible to PyTorch through its CUDA-compatible ROCm interface
- **Dataset:** `data/training_dataset_v2` (shared storage; not modified)
- **Class counts:** training 228 authorized / 228 unauthorized; validation 28 / 28; test 28 / 28
- **Input configuration:** 16 kHz sample rate, 512 FFT size, 160 hop length, 80 mel bins
- **Optimizer:** AdamW; weight decay 0.05
- **Batch size:** 8; **pretrained weights:** disabled; **ONNX export:** disabled
- **Default learning rate:** 0.0002; **epoch count:** 10 unless noted otherwise
- **Selection metric considered:** validation EER (lower is better)

## Results

Values shown are from the experiment summaries shared during testing. EER and accuracy are reported as percentages. The 1-epoch smoke test is included for context only.

| Experiment (topic) | Epochs | Seed | LR | Dropout | Val accuracy | Val EER ↓ | Test accuracy | Test EER ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `first-test` | 1 | 42 | 0.0002 | 0.0 | 71.4% | 28.6% | 66.1% | 28.6% |
| `experiment-10epochs` | 10 | 42 | 0.0002 | 0.0 | 85.7% | 14.3% | 82.1% | 14.3% |
| `dropout-02` | 10 | 42 | 0.0002 | 0.2 | 89.3% | 10.7% | 82.1% | 21.4% |
| `lr-0001` | 10 | 42 | 0.0001 | 0.0 | 82.1% | 17.9% | 80.4% | 21.4% |
| `baseline-seed123` | 10 | 123 | 0.0002 | 0.0 | 85.7% | 14.3% | 82.1% | 14.3% |
| `dropout-02-seed123` | 10 | 123 | 0.0002 | 0.2 | 85.7% | 14.3% | 82.1% | 10.7% |

For the four paired runs at seeds 42 and 123, mean validation EER was 14.3% without dropout and 12.5% with dropout. Mean test EER was 14.3% without dropout and 16.1% with dropout. These descriptive averages over only two seeds are **not** evidence of a reliable improvement.

## Observations

1. **The training pipeline works.** ConvNeXt Tiny trained and produced checkpoints, predictions, history, and evaluation metrics on the GPU.
2. **Ten epochs improved performance relative to the single-epoch smoke test.** Test accuracy increased from 66.1% to 82.1%, and test EER decreased from 28.6% to 14.3% in the baseline seed-42 comparison.
3. **Generalization is an issue.** Training loss approached zero, while validation loss fluctuated and often rose in later epochs. This is consistent with overfitting, although further checks are needed.
4. **Dropout had mixed outcomes.** At seed 42, dropout 0.2 improved validation EER (14.3% to 10.7%) but worsened test EER (14.3% to 21.4%). At seed 123, validation EER was equal (14.3%) while test EER was lower with dropout (10.7% versus 14.3%).
5. **Lower learning rate was not better in the one run tested.** Changing LR from 0.0002 to 0.0001 increased validation EER from 14.3% to 17.9% at seed 42.
6. **Operational error rates remain material.** In the baseline ten-epoch run, test FAR was 25.0% and test FRR was 10.7% at the evaluated decision threshold. These are not suitable evidence for real-world access-control deployment.

## Interpretation and limitations

- The validation and test splits each contain only 56 examples (28 per class), so rates change appreciably with a few predictions.
- Results come from exploratory comparisons, not a statistically powered benchmark. The same test set has been inspected during iteration; further model selection should use validation data only, with a fresh untouched holdout for a final estimate.
- Although project code passes a `dropout` argument into a classifier head, the full ConvNeXt-specific execution path was not verified in this analysis. Avoid claiming a causal dropout effect until that check is completed.
- Epoch-by-epoch validation accuracy and final summary accuracy may differ because of checkpoint selection or threshold handling; this detail has not been audited.
- Data leakage / speaker-disjointness between dataset splits has not yet been checked.
- No repeated trials beyond seeds 42 and 123 were performed for the paired dropout comparison.

## Suggested follow-up

1. Verify speaker- and recording-level separation across train, validation, and test.
2. Confirm checkpoint selection, threshold calibration, and the application of dropout within the ConvNeXt execution path.
3. If continued experimentation is warranted, compare configurations across additional seeds while selecting on validation EER and monitoring FAR at a deployment-relevant operating point.
4. Assess the chosen model once against a fresh, untouched holdout dataset before making deployment claims.

## Reproducing and reviewing results

Run outputs are stored under `runs/testingConvNext/<topic>/<run-id>/` on the Coder workspace. Each run may include `metrics.json`, `history.csv`, `best.pt`, `last.pt`, and prediction artifacts.

To regenerate the aggregate comparison locally in the workspace:

```bash
uv run iml-compare runs/testingConvNext --out convnext_comparison
```

The generated `convnext_comparison/` directory contains `table.csv`, `table.md`, `loss.png`, and `val_metrics.png` in the observed setup. These plots can be copied to `reports/convnext/plots/` for sharing; review the generated content before committing. The shared `data/` directory and large model checkpoints should not be included in the report.
