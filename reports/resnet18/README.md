# ResNet-18 — ImageNet fine-tuning

## Setup and run IDs

This report compares **four fresh fine-tuning runs**, FT1–FT4, selected by validation loss. Every run loads ImageNet weights
and trains the full ResNet-18, including a new two-class head. This experiment does not train
from scratch. All four runs complete 20 epochs with `checkpoint_metric="loss"` and
`early_stopping_patience=0`: the custom equivalent of `monitor="val_loss", mode="min"`.
These are fresh runs in `runs/petr/resnet18_val_loss/`; earlier EER-selected runs remain separate.

- Dataset: `training_dataset_v2`, **513 train / 63 validation / 63 test** clips.
  Validation and test each contain 28 allowed and 35 not-allowed clips.
- Input: 3 seconds, 16 kHz, 80 log-mel bins, FFT 512, hop 160, per-sample normalization.
- Fixed settings: Adam, batch 32, weight decay 0, seed 42, no scheduler, no early stopping.
  All four runs train for 20 epochs on an AMD Radeon 8060S / ROCm. No augmentation is active.
- **FT1:** LR 0.001, dropout 0; minimum validation loss **0.0032893** at **epoch 9**, EER 0%.
- **FT2:** LR 0.001, dropout 0.2; minimum validation loss **0.0057680** at **epoch 14**, EER 0%.
- **FT3:** LR 0.0003, dropout 0; minimum validation loss **0.0004675** at **epoch 20**, EER 0%.
- **FT4:** LR 0.0003, dropout 0.2; minimum validation loss **0.0004745** at **epoch 16**, EER 0%.

Initialization is delegated to `src/iml/models/__init__.py`: the ImageNet backbone is preserved,
its RGB stem is folded to one channel, and the new two-class Linear head uses He/Kaiming-normal
initialization (`fan_in`, ReLU), with zero bias. No layers are frozen. This is called by the
shared `fit()` loop; the script does not implement another model or training algorithm.

## Learning curves

![ResNet-18 fine-tuning: train loss, validation loss and validation EER for every run](plots/learning_curves.png)

The layout follows Artem's architecture report: **one panel per metric**, consistent run colors
and markers, one bottom legend, and rings marking minimum-validation-loss checkpoints. Every line is one run;
its 20 points are epochs, not 20 independent runs. Both loss panels share one symlog y scale: linear below 0.01, logarithmic above. This keeps
large early spikes and small later losses visible without removing zero values.
Training loss is measured during weight updates; validation loss uses the end-of-epoch model
in evaluation mode. Overlapping EER lines mean equal EER values, not identical predictions.

ImageNet fine-tuning separates this validation set quickly. Zero EER persists across many
epochs, rather than appearing at only one isolated lucky checkpoint. This does not establish
reliable generalization or a dropout advantage: each setting has only one seed.

## Checkpoint selection and results

The experiment minimizes **validation loss**, independently of EER. EER remains a reported
metric and cannot lock checkpoint selection at its first zero. Early stopping is disabled:
`last.pt` is epoch 20 for every run, while `best.pt` is the actual loss minimum in its history.
The same configured loss criterion would control early stopping if patience were enabled.
The shared library keeps EER selection as its default for other experiments and preserves
historical EER run names. Loss-selected configs receive distinct run names and cannot reuse
old EER-selected checkpoints.

The four loss-selected epochs are **9, 14, 20 and 16**; the first best-EER epochs in these same
histories are **3, 2, 3 and 5**. The reproducer checks every best epoch against `val_loss.idxmin()`.
Across configurations, selection now also minimizes checkpoint validation loss, with run ID
only as a tie-break. No test metric participates in either selection.

This selects **FT3, epoch 20**: LR 0.0003, dropout 0, validation loss
**0.0004675**. After loading these weights, the decision threshold is calibrated on
validation scores using the existing EER-point rule: **0.495224**. Changing the checkpoint
criterion does not fit a threshold on test. At that one fixed validation threshold:

- **Train:** accuracy 100.00%, FAR 0.00%, FRR 0.00%.
  The shared training loader drops its last partial batch; this evaluation has 512 clips.
- **Validation:** accuracy 100.00%, FAR 0.00%, FRR 0.00%, EER 0.00%.
- **Test:** accuracy **87.30%**, FAR **14.29% (5/35)**, FRR **10.71% (3/28)**.
  All five false accepts are T clips; the false rejects are one MS, one M and one P clip.

![Selected checkpoint: validation and test scores at one fixed threshold](plots/score_distributions.png)

The largest negative validation score is **0.004322**, and the smallest positive score
is **0.986125**. They remain separated, so validation EER is still zero even after
selecting a much later checkpoint by loss. Test scores still overlap. This change fixes the
checkpoint-selection objective; it does not establish that early EER selection caused the gap
or that microphone/session overfitting is its proven cause. These are fresh GPU runs, not a
strict paired causal comparison with the previous retained checkpoints.

EER is a per-split score-separation metric, not a joint train/test error rate. The shared
nearest-ROC-point estimate can be affected by rounding at tied candidates; independent
interpolation also confirmed the zero EERs in the previous fine-tuning audit.

## Why validation may look too optimistic

**Confirmed checks:** the saved metrics agree with independently recomputed confusion counts
and ROC candidates. No cross-split recording-ID overlap or exact PCM duplicates were found.
A further normalized waveform-correlation check of 4,572 same-speaker cross-split clip pairs
found no correlation >= 0.98 over an overlap of at least one second, allowing time shifts.
This check does not exclude all related sessions, re-encodings or other hidden dependencies.
Before any fine-tuning, the ImageNet model with its new random head had validation EER **31.79%**
at seed 42; the observed zero is not evidence that the classifier needed no training.

**Limited recording coverage:** T is labeled not allowed (0). Its 7 validation clips come from
**one** source recording, versus 10 train recordings and 2 test recordings. The full validation
set contains **63 clips**, not seven. Several clips from one recording do not provide the same
coverage of devices, rooms and sessions as independent recordings. Documented phone/codec/source
differences could also let the model exploit the recording channel rather than stable voice cues;
we have not established that this is the cause of the errors.

**Model-selection optimism:** validation does not contribute gradients, but it selects epochs by loss,
hyperparameters and the decision threshold. Repeatedly choosing the best result on the same finite
validation set can favor an accidentally optimistic estimate; this is overfitting through model
selection, not proof that train files entered validation ([Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).
It is a plausible contributor, not a diagnosed cause here. Persistent zero EER across many epochs
means the result cannot be explained solely as one lucky checkpoint. The validation/test gap may
combine limited coverage, channel differences, training overfitting and selection optimism.

**Comparison with Michal:** his published ConvNeXt report used 456/56/56 clips and training from
scratch. These are different experimental conditions, so its nonzero EER does not establish a
ResNet-specific bug or make either architecture better. More recent matched runs would need
matching dataset fingerprints and configs before comparison.

The next evaluation should cover independent recording sessions/devices for T and the allowed
speakers and reserve a fresh holdout. Do not randomly distribute neighboring excerpts from one
recording across splits. The current test set has already been inspected; it is exploratory and
cannot serve as an untouched final benchmark. These results do not establish deployment readiness.

## Reproduce

```bash
uv sync --group dev
uv run iml-data validate data/training_dataset_v2
uv run python analyze_resnet18.py --train
```

Only missing loss-monitored ImageNet fine-tuning variants are trained; matching finished runs are reused.
Omit `--train` to audit existing predictions and regenerate the two figures without training.
Dataset SHA1: `10ac1fb033e3b1845bc52ea2e2229c2982ad11f0`.
Configs, histories, audit receipts, predictions and weights stay in `runs/petr/resnet18_val_loss/`
on Coder, outside Git. MLflow receives quality metrics only when reachable; it was unreachable
for these runs, so they were not tracked there. The selected ONNX export matches PyTorch
(max absolute output difference 5.72e-6). Only this README, the two figures and the script are published.

Validation: 69 core tests passed, including late loss-minimum checkpoint retention, consistent
early-stopping monitoring and compatibility of historical EER run names. All four new
checkpoints were verified against their histories and all last checkpoints are at epoch 20.
