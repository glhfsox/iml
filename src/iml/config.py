"""Single experiment configuration object shared by notebooks, scripts and the app.

Every knob that an experiment may vary lives here, so that a run is fully described by
one JSON file and its folder name is derived from the *differences* from the defaults.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

MODEL_NAMES = (
    "resnet18",
    "efficientnetv2_s",
    "convnext_tiny",
    "resnest26d",
    "regnet_y_800mf",
    "custom",
)

# Architecture knobs of the old hand-written models; old checkpoints may still carry them.
_LEGACY_KEYS = {"width", "depth", "bn_position", "activation"}

# Fields that do not change the experiment itself and are excluded from the run name/hash.
_BOOKKEEPING = {"member", "topic", "tags", "notes", "num_workers", "export_onnx"}


@dataclass
class ExperimentConfig:
    """All hyper-parameters of one training run.

    Attributes are grouped by what they control; the dataclass defaults are the ``resnet18``
    baseline. Per-model baselines live in ``app/models/configs/<model>.json`` (``make configs``).
    All models are ImageNet CNNs sharing the same knobs: ``pretrained`` (ImageNet weights),
    ``init`` (applied only when training from scratch; ``build_model`` rejects a non-default
    ``init`` together with ``pretrained=True``) and ``dropout`` (before the final classifier).
    ``model="custom"`` is assembled from ``arch`` (``iml.models.spec.build_from_spec``)
    and needs ``pretrained=False``.
    """

    # --- identity / bookkeeping ---
    member: str = "anon"  # notebooks/<member>/
    topic: str = "default"  # comparisons/<topic>/
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # --- model ---
    model: str = "resnet18"  # one of MODEL_NAMES
    init: str = "he"  # he | xavier | uniform | default; only when pretrained=False
    dropout: float = 0.0  # dropout before the final classifier (stochastic depth not exposed)
    pretrained: bool = True  # ImageNet weights (transfer learning); False = from scratch
    # model="custom" only: declarative block list, see iml/models/README.md.
    # custom has no ImageNet weights, so its configs must set pretrained=false explicitly.
    arch: list[dict[str, Any]] = field(default_factory=list)

    # --- input / spectrogram ---
    sample_rate: int = 16000
    excerpt_sec: float = 3.0  # length of one excerpt (= the 3 s clips of DATA_CONTRACT.md)
    n_fft: int = 512
    hop_length: int = 160  # 10 ms at 16 kHz
    n_mels: int = 80
    normalize_input: str = "per_sample"  # none | per_sample | global
    augment: dict[str, Any] = field(default_factory=dict)  # see iml.data.SpectrogramDataset (later)

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
    checkpoint_metric: str = "eer"  # eer | loss; minimize on validation for checkpoint/early stop
    mc_dropout_samples: int = 0  # >0: Monte-Carlo dropout with that many stochastic passes
    export_onnx: bool = True  # write + verify <run>/model.onnx at the end of fit (iml.export)

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"unknown model {self.model!r}; expected one of {MODEL_NAMES}")
        if not isinstance(self.arch, list | tuple):
            raise ValueError(f"arch must be a list of layer dicts, got {type(self.arch).__name__}")
        self.arch = copy.deepcopy(list(self.arch))  # never alias a caller's (shared) spec
        if self.model == "custom" and not self.arch:
            raise ValueError('model="custom" needs a non-empty arch (list of layer dicts)')
        if self.model != "custom" and self.arch:
            raise ValueError(f"arch is only used by model='custom'; {self.model} needs arch=[]")
        if self.checkpoint_metric not in {"eer", "loss"}:
            raise ValueError("checkpoint_metric must be eer or loss")
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
        legacy = sorted(_LEGACY_KEYS & set(d))
        if legacy:
            warnings.warn(
                f"dropping legacy config keys {legacy} (removed model knobs)", stacklevel=2
            )
            d = {k: v for k, v in d.items() if k not in _LEGACY_KEYS}
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
        default = ExperimentConfig(model=self.model, arch=self.arch).to_dict()
        default["arch"] = []  # a custom arch always counts as a difference
        mine = self.to_dict()
        return {
            k: v
            for k, v in mine.items()
            if k not in _BOOKKEEPING and k != "model" and v != default[k]
        }

    def config_hash(self, n: int = 6) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k not in _BOOKKEEPING}
        # Preserve names of historical EER-selected runs; loss selection is a new experiment.
        if self.checkpoint_metric == "eer":
            payload.pop("checkpoint_metric")
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:n]

    def run_name(self, max_slug: int = 60) -> str:
        """``<model>_<slug of non-default fields>_<hash>``, e.g. ``resnet18_lr0.01_3f9a1c``.

        The hash guarantees uniqueness; the slug is only there to make folders readable.
        """
        parts = []
        for k, v in sorted(self.diff_from_default().items()):
            if isinstance(v, dict | list):
                v = "x"  # content is in config.json; keep the name short
            parts.append(f"{k.replace('_', '-')}{'-' if isinstance(v, str) else ''}{v}")
        slug = "_".join(parts)[:max_slug].rstrip("_-")
        return "_".join(p for p in (self.model, slug, self.config_hash()) if p)
