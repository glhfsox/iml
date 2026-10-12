import pytest
import torch

from app.main import load_model, predict
from app.models import model_dir, pipeline_module
from app.models.speaker_verification.contracts import Verdict
from iml.cli._data import load_splits
from iml.config import MODEL_NAMES, ExperimentConfig
from iml.data.audio import save_wav
from iml.models import SUPPORTED_KNOBS, build_model
from iml.train import fit


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_folder_builds_a_pipeline(name):
    assert (model_dir(name) / "pipeline" / "operators.py").exists()
    assert (model_dir(name) / "transform.py").exists()
    cfg = ExperimentConfig(model=name, **({"width": 8} if "width" in SUPPORTED_KNOBS[name] else {}))
    pipe = pipeline_module(name).build_pipeline(cfg, build_model(cfg), torch.device("cpu"))
    v = pipe(torch.randn(16000))
    assert isinstance(v, Verdict) and v.label in (0, 1) and 0.5 <= v.confidence <= 1.0


def test_app_loads_run_and_predicts(tmp_path, tiny_data):
    cfg = ExperimentConfig(
        width=4, epochs=1, batch_size=16, num_workers=0, dropout=0.2, mc_dropout_samples=4
    )
    rec = fit(
        cfg,
        load_splits(cfg, data_dir=tiny_data),
        root=tmp_path,
        device=torch.device("cpu"),
        verbose=False,
    )
    save_wav(tmp_path / "x.wav", torch.randn(3 * 16000) * 0.1, 16000)
    v = predict(rec.path, tmp_path / "x.wav")
    assert v.label in (0, 1) and v.uncertainty >= 0
    r = load_model(rec.best_checkpoint, device=torch.device("cpu"))
    assert r.cfg == cfg and r.predict_wave(torch.randn(8000)).label in (0, 1)


def test_ecapa_embed_operator():
    from app.models.speaker_verification.ecapa_tdnn.pipeline.operators import Embed

    cfg = ExperimentConfig(model="ecapa_tdnn", width=8)
    emb = Embed(build_model(cfg))(torch.randn(2, 1, 64, 100))
    assert emb.shape == (2, 192)


def test_app_uses_run_threshold_by_default(tmp_path, tiny_data):
    cfg = ExperimentConfig(width=4, epochs=1, batch_size=16, num_workers=0)
    rec = fit(
        cfg,
        load_splits(cfg, data_dir=tiny_data),
        root=tmp_path,
        device=torch.device("cpu"),
        verbose=False,
    )
    r = load_model(rec.path, device=torch.device("cpu"))
    assert r.pipeline.operators[-1].threshold == rec.metrics["decision_threshold"]
    assert (
        load_model(rec.path, device=torch.device("cpu"), threshold=0.9)
        .pipeline.operators[-1]
        .threshold
        == 0.9
    )
