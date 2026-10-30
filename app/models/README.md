# app/models

One folder per architecture under the task group `speaker_verification/`. The folder is the
unit that gets frozen and shipped: the pipeline code that turns a waveform into a `Verdict`
plus ONE artifact, `model.onnx`. The graph is self-describing: `iml.export.export_onnx` writes
the run's `ExperimentConfig` (`iml.config`, JSON), the val-chosen decision threshold
(`iml.decision_threshold`), the model name and the data contract version into the ONNX
`metadata_props`, so no `config.json` / `best.pt` is needed next to it.

```
speaker_verification/
  contracts.py              Verdict dataclass (output of every pipeline)
  resnet18/ efficientnetv2_s/ convnext_tiny/ resnest26d/ regnet_y_800mf/ custom/   one per iml.config.MODEL_NAMES
    pipeline/__init__.py    build_pipeline(cfg, model, device, threshold, mc_samples)
    pipeline/operators.py   operator chain: LogMelTransform -> make_infer(model) -> DecodeVerdict
                            (OnnxInfer for an onnxruntime session, TorchInfer for an nn.Module)
    transform.py            input contract of the model + its preprocessing operator
    model.onnx              the frozen model: graph with dynamic batch/time axes + config and
                            threshold in its metadata (written by fit, copied by make freeze, gitignored)
  wespeaker_resnet34/        pretrained zero-shot embedder: no config.json / best.pt, no make freeze
    pipeline/__init__.py    build_pipeline(bank, threshold=None, embedder=None)
    pipeline/operators.py   FbankTransform -> Embed (ONNX) -> BankScore (best centroid cosine) -> DecodeVerdict
    transform.py            16 kHz wave -> [T, 80] WeSpeaker fbank segments (PreprocCfg from the bank meta)
    embedder.py             WeSpeakerEmbedder (onnxruntime) + ensure_checkpoint (downloads if missing)
    voxceleb_resnet34_LM.onnx  upstream graph, name kept: banks record it in meta["checkpoint"] (gitignored)
utils.py                    shared operators + load_onnx() (production) / load_weights() (dev, best.pt)
```

Every folder starts from a **stable pretrained graph**: `make init-models` writes
`iml-export --pretrained <model>` into each folder — the ImageNet backbone with a fresh 2-class
head (seeded), threshold 0.5 and the model's baseline config in the metadata. It is the fallback
the app can always run, and the weights every fine-tuning run starts from (`pretrained=true` is the
config default, so `fit` begins from the same backbone). `custom` has no pretrained weights and gets
its `model.onnx` only through `make freeze`.

```bash
make init-models                                     # 5 x model.onnx from ImageNet weights (downloads once)
```

Freeze a trained run into its model folder (the folder is chosen from the run's `config.json`; the
run's `model.onnx` is exported first if it has none) and run it:

```bash
make freeze RUN=runs/<member>/<topic>/<run_name>     # copies only model.onnx
uv run python -m app.main --model resnet18 --wav recording.wav             # frozen graph, onnxruntime
uv run python -m app.main --run runs/<member>/<topic>/<run_name> --wav recording.wav  # PyTorch best.pt (dev)
uv run python -m app.main --run path/to/model.onnx --wav recording.wav     # any exported graph
```

The ONNX path runs on CPU (the log-mel front end stays in PyTorch, only the classifier runs in
onnxruntime) and has no MC dropout: the verdict's `uncertainty` is 0, and asking for
`mc_samples` raises. Use `--run` on the run folder for MC-dropout estimates.

The pretrained `wespeaker_resnet34` is scored against an embedding bank (`.npz` saved by
`Gallery.save`, threshold in its meta); notebooks/glhfsox/speaker_id re-exports its
`transform.py` / `embedder.py`:

```bash
uv run python -m app.main --model wespeaker_resnet34 --bank notebooks/glhfsox/bankv3.npz --wav rec.wav
```

Shared code lives in `utils.py`; a model folder only composes it (and adds its own operators
when it needs something the others do not, e.g. the bank scoring of `wespeaker_resnet34`).

The six trainable folders are resnet18 (torchvision ResNet-18), efficientnetv2_s, convnext_tiny,
resnest26d, regnet_y_800mf (timm backbones) and custom; wespeaker_resnet34 is the pretrained one.
Custom models are assembled from the block library `src/iml/models/blocks.py` (one module; see
`src/iml/models/README.md`); an architecture added to `MODEL_NAMES` gets its own folder here,
copied from `resnet18/` with the import paths changed.
