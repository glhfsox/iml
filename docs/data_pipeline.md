# Telegram data preparation pipeline

The repository contains the collector and preparation code but never stores Telegram credentials
or real voice recordings in Git. Collection and manual review happen locally; `data/` remains
ignored.

## Data flow

1. `voice-dataset` downloads messages only from explicitly selected chats and dates. It keeps the
   original media and produces mono 16 kHz PCM16 WAV files. A Telegram Desktop JSON export is an
   alternative source.
2. Automatic quality checks route recordings to `accept`, `review` or `reject`.  They do not
   identify the speaker and cannot reliably detect music or another person.
3. A human reviews the label sheet.  Training requires one verified speaker, explicit consent,
   a stable `speaker_ref`, a `session_ref` and `manual_review_status=approved`.
4. `iml-prepare-data` validates the reviewed metadata, derives one-vs-rest labels, creates a
   leakage-safe split, copies normalized WAV files into a gitignored output directory and writes
   a dataset fingerprint.
5. `iml-train` loads `splits.json`.  `SpectrogramDataset` crops a waveform and creates the
   log-mel spectrogram at runtime; the numeric label travels beside the spectrogram.

## Required collector fields

The input CSV must contain the fields validated in `iml.data.manifest.REQUIRED_METADATA_FIELDS`.
Rows are eligible only when they represent a non-forwarded voice message with:

- `dataset_status=ready`;
- `manual_review_status=approved`;
- `speaker_status=verified_single`;
- `consent_status=confirmed_self` or `confirmed_other`;
- `quality_decision=accept` or a manually approved `review`;
- `dedupe_status=unique`;
- a non-empty `speaker_ref`, `session_ref` and normalized audio path.

The audio must be a readable mono 16 kHz PCM16 WAV between 2 and 45 seconds.

## Preparing one owner model

The base dataset stores speaker identity rather than a permanent target/impostor role.  Choose
the owner when preparing a model:

```bash
uv run iml-prepare-data \
  --metadata /private/collector/data/metadata.csv \
  --source-root /private/collector \
  --target-speaker speaker_01 \
  --output data/processed/speaker_01
```

Run the command again with another `--target-speaker` and output directory to train a separate
one-vs-rest model for another participant.

The output contains:

```text
data/processed/speaker_01/
  audio/                  normalized WAV files, never committed
  manifest.csv            minimal pseudonymous row-level manifest
  splits.json             exact train/val/test membership consumed by training
  dataset_report.json     counts, duration, exclusions and split policy
```

Every participant is divided by whole `(speaker_ref, session_ref)` groups. All participants are
therefore represented in train, validation and test, while one recording session cannot leak
between them. Every split must contain both classes. Each participant needs at least three
independent sessions; with four people this produces one target and three impostors per model.

This is a closed-participant university experiment. It does not by itself prove rejection of an
unknown outsider. A later open-set evaluation should reserve additional consenting speakers who
never occur in training.

## Training

```bash
uv run iml-train \
  --config configs/cnn10.json \
  --data-manifest data/processed/speaker_01/splits.json \
  --member <name> \
  --topic baseline
```

`iml-train` injects the manifest fingerprint into `ExperimentConfig`, so the run directory and
config hash change when the dataset changes.  It also copies `splits.json` into the finished run
folder beside metrics and checkpoints.

## Privacy and Git

Do not force-add `data/`, audio, Telegram session files, SQLite state, API credentials or the
review workbook.  Transfer the reviewed dataset to the training server through an approved
private channel and mount it into the Docker training container as a volume.

## Local collection commands

```bash
uv sync --group dev --group data
uv run voice-dataset doctor
uv run voice-dataset rank-chats --start 2025-10-01 --end 2026-10-07
uv run voice-dataset collect-telegram \
  --start 2025-10-01 --end 2026-10-07 \
  --expected-chat-count 3 --limit 20
```

The first collection must be a 20-message pilot. A later batch uses `--next-batch --limit N` and
the same stable `--chat-ref` values. Ordinary voice messages are the default; video notes are
opt-in. Login values are requested in the local terminal with hidden input, and the session is
stored at `data/private/telegram.session`.

The review CSV includes the WAV path, date, duration and quality fields. It suggests
`speaker_ref` from Telegram's pseudonymous sender and `session_ref` from the recording day, but
keeps `speaker_status=unreviewed` and `manual_review_status=pending`.

For shorter local codes, create the ignored private file
`data/private/speaker_aliases.csv` with columns `chat_ref,sender_ref,speaker_ref`. A row with
`chat_ref=*` assigns a code to the same account in every chat; a row with `sender_ref=*` assigns
the other participant's code for one direct chat. These values are only review hints. Forwarded
messages receive no speaker hint, and every row still requires listening and explicit approval.

After listening, correct the speaker suggestion if needed and approve or reject each row. Leave
`access_role=unassigned`: `iml-prepare-data --target-speaker ...` derives target/impostor labels
later for each one-vs-rest model. Then run:

```bash
uv run voice-dataset apply-labels data/labels/speaker_labels_<run>.csv
```

Automatic quality checks cannot prove who is speaking or reliably distinguish speech from music,
so they never replace this review.
