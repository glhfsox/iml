import pytest
import torch

pytest.importorskip("onnxruntime")

from iml.cli.export import check_onnx, export_onnx
from iml.config import MODEL_NAMES, ExperimentConfig
from iml.data.dataset import DummyDataset, make_loaders
from iml.models import SUPPORTED_KNOBS
from iml.train import fit


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_export_every_model_matches_pytorch(tmp_path, name):
    width = {"width": 8} if "width" in SUPPORTED_KNOBS[name] else {}
    cfg = ExperimentConfig(
        model=name, **width, epochs=1, batch_size=16, num_workers=0, member="t", topic="onnx"
    )
    rec = fit(
        cfg,
        make_loaders({"train": DummyDataset(cfg, n=16)}, cfg),
        root=tmp_path,
        device=torch.device("cpu"),
        verbose=False,
    )
    out = export_onnx(rec.path)
    assert out.exists()
    assert check_onnx(out, rec.path) < 1e-3
