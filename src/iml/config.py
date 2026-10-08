"""Single experiment configuration object shared by notebooks, scripts and the app.

Every knob that an experiment may vary lives here, so that a run is fully described by
one JSON file and its folder name is derived from the *differences* from the defaults.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

MODEL_NAMES = ("simple_cnn", "cnn10", "resnet18", "mobilenet_v2", "ecapa_tdnn")

# Fields that do not change the experiment itself and are excluded from the run name/hash.
_BOOKKEEPING = {"member", "topic", "tags", "notes", "num_workers"}


@dataclass
class ExperimentConfig:
    """All hyper-parameters of one training run.

    Attributes are grouped by what they control; the dataclass defaults are the
    ``simple_cnn`` baseline. Per-model baselines (e.g. ``depth=4`` for cnn10) live in
    ``configs/<model>.json`` (``make configs``). Knobs a model does not use must stay at
    their default, otherwise ``build_model`` refuses the config (no fake comparisons).
    """

    # --- identity / bookkeeping ---
    member: str = "anon"  # notebooks/<member>/
    topic: str = "default"  # comparisons/<topic>/
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # --- model ---
    model: str = "simple_cnn"  # one of MODEL_NAMES
    width: int = 32  # base number of channels
    depth: int = 2  # number of conv blocks (simple_cnn: 1-4, cnn10: 2-6)
    bn_position: str = "before"  # before | after | none (relative to the activation)
    activation: str = "relu"  # relu | sigmoid | gelu
    init: str = "he"  # he | xavier | uniform | default
    dropout: float = 0.0
    pretrained: bool = False  # resnet18 / mobilenet_v2 only: ImageNet weights (transfer learning)

    # --- input / spectrogram ---
    sample_rate: int = 16000
    excerpt_sec: float = 2.0  # length of one training excerpt
    n_fft: int = 512
    hop_length: int = 160  # 10 ms at 16 kHz
    n_mels: int = 64
    normalize_input: str = "per_sample"  # none | per_sample | global
    augment: dict[str, Any] = field(default_factory=dict)  # see iml.data.augment (later)

    # --- optimisation ---
    optimizer: str = "adam"  # sgd | adam | adamw
    lr: float = 1e-3
    momentum: float = 0.9  # sgd only
    weight_decay: float = 0.0
    class_weight: str = "none"  # none | balanced (inverse-frequency CE weights from the train set)
    scheduler: str = "none"  # none | cosine | step
    batch_size: int = 32
    epochs: int = 20
    early_stopping_patience: int = 0  # 0 = off; counts epochs without val improvement
    grad_clip: float = 0.0  # 0 = off
    amp: bool = True  # mixed precision on the GPU (HIP/ROCm or CUDA); ignored on MPS/CPU
    torch_compile: bool = (
        False  # torch.compile the model before training (Linux GPU: HIP/ROCm or CUDA)
    )
    seed: int = 42
    num_workers: int = 2

    # --- evaluation ---
    mc_dropout_samples: int = 0  # >0: Monte-Carlo dropout with that many stochastic passes

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"unknown model {self.model!r}; expected one of {MODEL_NAMES}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.mc_dropout_samples > 0 and self.dropout <= 0:
            raise ValueError(
                "mc_dropout_samples > 0 needs dropout > 0, otherwise MC dropout is a no-op"
            )

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def with_overrides(self, *overrides: str) -> ExperimentConfig:
        """Return a copy with ``key=value`` overrides applied (CLI style).

        Non-string fields are parsed as JSON (``lr=0.01``, ``pretrained=true``,
        ``tags=["a","b"]``); string fields are taken verbatim (``optimizer=sgd``, ``topic=1``).
        """
        d = self.to_dict()
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must look like key=value, got {item!r}")
            key, raw = item.split("=", 1)
            if key not in d:
                raise ValueError(f"unknown config key {key!r}")
            if isinstance(d[key], str):
                value: Any = raw  # string fields are taken verbatim (member=123 stays "123")
            else:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"cannot parse {item!r} for field {key!r}: {e}") from e
            d[key] = value
        return ExperimentConfig.from_dict(d)

    # ------------------------------------------------------------------ naming
    def diff_from_default(self) -> dict[str, Any]:
        """Experiment-relevant fields whose value differs from the default."""
        default = ExperimentConfig(model=self.model).to_dict()
        mine = self.to_dict()
        return {
            k: v
            for k, v in mine.items()
            if k not in _BOOKKEEPING and k != "model" and v != default[k]
        }

    def config_hash(self, n: int = 6) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k not in _BOOKKEEPING}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:n]

    def run_name(self, max_slug: int = 60) -> str:
        """``<model>_<slug of non-default fields>_<hash>``, e.g. ``simple_cnn_lr0.01_3f9a1c``.

        The hash guarantees uniqueness; the slug is only there to make folders readable.
        """
        parts = []
        for k, v in sorted(self.diff_from_default().items()):
            if isinstance(v, dict | list):
                v = "x"  # content is in config.json; keep the name short
            parts.append(f"{k.replace('_', '-')}{'-' if isinstance(v, str) else ''}{v}")
        slug = "_".join(parts)[:max_slug].rstrip("_-")
        return "_".join(p for p in (self.model, slug, self.config_hash()) if p)
