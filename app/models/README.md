# app/models

One folder per architecture under the task group `speaker_verification/`. The folder is the
unit that gets frozen and shipped: weights, exported ONNX graph, the config they were trained
with, and the pipeline that turns a waveform into a `Verdict`.

```
speaker_verification/
  contracts.py              Verdict dataclass (output of every pipeline)
  simple_cnn/ cnn10/ resnet18/ mobilenet_v2/ ecapa_tdnn/
    pipeline/__init__.py    build_pipeline(cfg, model, device, threshold, mc_samples)
    pipeline/operators.py   operator chain: LogMelTransform -> TorchInfer -> DecodeVerdict
    transform.py            input contract of the model + its preprocessing operator
    config.json             ExperimentConfig the frozen weights were trained with   (make freeze)
    best.pt                 frozen weights                                          (make freeze, gitignored)
    model.onnx              exported graph with dynamic batch/time axes              (make freeze, gitignored)
utils.py                    shared operators + load_weights()
```

Freeze a run into its model folder (the folder is chosen from the run's `config.json`):

```bash
make freeze RUN=runs/<member>/<topic>/<run_name>
uv run python -m app.main --model cnn10 --wav recording.wav
```

Shared code lives in `utils.py`; a model folder only composes it (and adds its own operators
when it needs something the others do not, e.g. an embedding extractor for ECAPA-TDNN).
