# Understanding the recording dataset

This notebook helps answer: **Is the data balanced? How many different recordings do the
clips represent? How does audio quality vary? What does the model receive?**

It performs descriptive analysis: counts, distributions, audio-level measurements, duplicate
checks, and visual inspection. The notebook works with v2,
v3, v4, or another dataset in the same format. **The findings below cover v2 only.**

## Terms used here

- A **clip** is a three-second excerpt. Several clips can come from one recording and may overlap.
- A **recording ID** identifies a source recording. Different IDs can still share a room,
  microphone, or session; they are not proof of independent recording conditions.
- **Allowed / 1** means the person may enter; **invalid / 0** means they should be rejected.
  “Invalid” is an access label, not an audio-quality judgment.
- **M, P, MS, D** are four allowed people; **T** is one invalid person. **External N** combines
  40 other invalid speakers, not one person.
- **Training** teaches the model; **validation** helps choose settings; **test** assesses the
  chosen model. Avoid repeatedly using test plots or results to select model settings.

**Important:** “before/after normalization” compares features computed from the **same stored,
already-processed clip**. It does not compare original audio with processed audio. No matched
original v2 recordings were available, so these plots cannot establish how preprocessing
changed v2's audio level. Waveform measurements describe only the stored clips.

## Running the notebook

Open [recordings.ipynb](recordings.ipynb) with the `iml` environment. Select `DATASET_NAME`
in the configuration cell, then run all cells:

```python
DATASET_NAME = "training_dataset_v2"  # or training_dataset_v3 / training_dataset_v4
```

The existing setting also accepts an `IML_EDA_DATASET` environment override; the notebook
prints the actual input path. Absolute dataset paths work too. CSVs determine the labels;
matching provenance files or documented filename conventions supply recording/speaker IDs.

Results are kept separately for each dataset:

```text
notebooks/spdsslg/eda/
  DATA_ANALYSIS.md
  recordings.ipynb
  reports/
    training_dataset_v2/
    training_dataset_v3/
    training_dataset_v4/
```

Each report folder contains five figures, summary tables, per-clip measurements,
`observations.md`, and metadata recording the settings, software versions, seed, and input
checksum. The checksum confirms that input data stayed unchanged. The v2 PNG figures and
`observations.md` are included as the example report. Reports for v3/v4 are local and
gitignored; JSON metadata and CSV files are also ignored. Tables and per-clip measurements
are displayed in the notebook. The figure links below point to the v2 example report.

## v2 at a glance

| Split | Allowed clips | Invalid clips | Total |
|---|---:|---:|---:|
| Training | 228 | 285 | 513 |
| Validation | 28 | 35 | 63 |
| Test | 28 | 35 | 63 |
| **All data** | **284** | **355** | **639** |

The split is approximately **80% / 10% / 10%**. These 639 clips come from **259 recording IDs**:
52 for allowed people and 207 for invalid people.

## Reading the figures

### 1. Class and group counts

[Open the figure](reports/training_dataset_v2/dataset_composition.png).

**Left:** bars count allowed/invalid clips in each split. **Right:** each bar represents a
person or source group; its colors show the training, validation, and test portions.

**v2 insight:** all splits have 1.25 invalid clips per allowed clip, a modest imbalance.
Every local person contributes 71 clips: 57 training, seven validation, seven test.
External N contributes 284 clips from 40 people. Equal clip counts for the allowed people
hide unequal recording variety, shown next.

### 2. Recording counts and the staircase graph

[Open the figure](reports/training_dataset_v2/recording_groups.png).

**Left:** bars count recording IDs rather than clips. **Right:** each staircase answers:
**“What percentage of this group's recordings have this many clips or fewer?”**

Choose a clip count on the horizontal axis, move up to a colored line, and read the percentage.
In older outputs labelled 0–1, 0.4 means 40%.

For **D**, the line reaches **40% at seven clips**: two of five recordings contribute seven
clips each. It stays flat until 19, then reaches 100%: the remaining three recordings each
contribute 19 clips. A flat section means no recordings have clip counts in that interval;
it is not a gap in recording time.

The percentage counts **recordings**, not clips. Those two seven-clip recordings are 40% of
D's recordings but contribute only 14/71, about 20%, of D's clips. Curves farther right mean
more excerpts per recording, not better model accuracy or audio quality.

| Person/source | Recording IDs | Largest number of clips from one recording |
|---|---:|---:|
| M | 12 | 13 |
| P | 30 | 6 |
| MS | 5 | 20 |
| D | 5 | 19 |
| T | 13 | 10 |
| External N | 194 | 6 |

**v2 insight:** P's 71 clips span 30 recordings, while D's and MS's each span only five.
Their equal clip counts therefore represent very different amounts of recording variety.
Closely related excerpts provide less independent evidence than the clip count suggests.

### 3. Audio levels and quality checks

[Open the figure](reports/training_dataset_v2/waveform_quality.png).

In each box plot, the line is the middle value and the box contains the middle half of the
clips. Dots mark values farther from the main range; they are not automatically bad audio.

- **Typical waveform level (RMS):** average signal strength. In dBFS, −16 is stronger than
  −26. This measures the stored digital signal, not perceived loudness or how loudly someone spoke.
- **Peaks:** the strongest moment in each clip. Zero dBFS is the digital ceiling.
- **Low-energy frames:** the fraction of short intervals below a fixed −45 dBFS threshold.
  It describes low-level audio; it is not a reliable silence/speech percentage.
- **PCM-limit samples:** the percentage of clips containing samples exactly at the digital
  minimum/maximum. These are inspection flags, not proof of audible clipping.

**v2 insights:** T has the highest median RMS (−16.3 dBFS), and M the lowest (−26.5 dBFS),
a **10.2 dB difference**. T also has fewer low-energy frames: median 3.3%, versus M's 20.7%.
Voice, microphone distance, background sound, and recording settings could all contribute.

**33/639 clips** contain PCM-limit samples: T has 18, M eight, P six, D one. That includes
about 25% of T's clips. Listen to these before deciding whether anything needs fixing.
These differences between stored groups do not establish a pre/post-processing level change.

### 4. Frequency profiles before and after input normalization

[Open the figure](reports/training_dataset_v2/spectral_profiles.png).

Lines summarize the strength of different frequency ranges. Low mel-band numbers represent
lower frequencies; high numbers represent higher frequencies. Each line is a group's median
profile. Shading covers the middle half of its clip profiles, not a confidence interval.

**Left:** log-mel features before normalization. **Right:** the same features rescaled to a
common numerical reference. The vertical units differ, so heights cannot be compared directly
as an audio-level change. No waveform is altered.

**v2 insights:** T's unnormalized profile is generally above M's, consistent with their signal
levels. Normalization brings the curves closer, but D still has relatively weak upper-frequency
content. It reduces broad scale differences without making all groups identical or removing
noise. The plots cannot tell whether remaining differences come from voices or recording
conditions, or whether the model uses them. But knowing our data, it is reasonable to assume that the difference comes not from voices but from recording conditions.

### 5. Waveforms and model inputs

[Open the figure](reports/training_dataset_v2/waveforms_and_model_inputs.png).

Each row shows a clip close to its group's median signal level, identified in the notebook.
**Left:** amplitude over three seconds. **Middle:** the spectrogram before input normalization.
**Right:** the normalized spectrogram supplied to the model. Time runs left to right; frequency
bands run bottom to top. Lighter areas indicate stronger values within that panel's scale.

**v2 insight:** T's example has stronger waveform excursions than M's, consistent with the
all-clip statistics. Both spectrogram columns retain gaps and frequency patterns: normalization
rescales features rather than cleaning audio. Their units differ, so brightness cannot measure
a loudness change. These are representative examples, not every clip in each group.

Pictures share fixed color limits within each mode: raw log-mel uses −80 to 40 dB,
`global` uses −1 to 2 (the same dB range after its scaling), and `per_sample` uses −3 to 5.
This makes colors comparable across images and datasets using the same mode. Values outside
these ranges use the endpoint colors; the underlying tensors are unchanged. Different modes
still use different units, so colors are not directly comparable between modes.

## General observations from v2

- **Recording coverage is uneven.** The four allowed people have 284 clips but only 52 recording
  IDs, particularly few for D and MS. New independent sessions would add variation that repeated
  crops cannot supply.
- **Validation has limited independent evidence.** Its 28 allowed clips come from six recording
  IDs: one each for M/MS/D and three for P. One additional false rejection moves its rejection
  error rate by about 3.6 percentage points; one false acceptance moves its acceptance error
  rate by about 2.9 points.
- **Training reuses audio.** There are 148 overlapping adjacent clip pairs within training
  recordings, and none in validation/test by their known offsets. The summed 31.95 minutes
  of clips are not 31.95 minutes of independent source audio.
- **Basic separation checks pass.** No recording ID crosses splits, and no clips have identical
  waveform hashes. Similar excerpts, shared sessions, or shared recording conditions remain possible.
- **Visible differences may offer shortcuts.** The model could learn recording conditions as
  well as voice characteristics. These plots identify that possibility and we need to account for it!