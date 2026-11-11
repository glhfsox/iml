import json

import pytest

from iml.config import ExperimentConfig


def test_json_roundtrip(tmp_path):
    cfg = ExperimentConfig(model="efficientnetv2_s", lr=0.01, augment={"time_mask": 20}, tags=["a"])
    cfg.to_json(tmp_path / "c.json")
    assert ExperimentConfig.from_json(tmp_path / "c.json") == cfg


def test_run_name_is_stable_and_readable():
    a = ExperimentConfig(model="convnext_tiny", dropout=0.2, lr=0.01)
    b = ExperimentConfig(model="convnext_tiny", dropout=0.2, lr=0.01, member="x", topic="y")
    assert a.run_name() == b.run_name()  # bookkeeping differs
    assert a.run_name().startswith("convnext_tiny_dropout0.2_lr0.01_")
    assert a.run_name() != ExperimentConfig(model="convnext_tiny", dropout=0.2, lr=0.02).run_name()


def test_default_run_name_has_no_slug():
    name = ExperimentConfig().run_name()
    assert name.count("_") == 1  # resnet18 + hash only
    assert name.startswith("resnet18_")


def test_overrides_parse_types():
    cfg = ExperimentConfig().with_overrides(
        "lr=0.01", "pretrained=true", "optimizer=sgd", 'tags=["x"]'
    )
    assert (
        cfg.lr == 0.01 and cfg.pretrained is True and cfg.optimizer == "sgd" and cfg.tags == ["x"]
    )


def test_string_fields_are_not_json_parsed():
    cfg = ExperimentConfig().with_overrides("member=123", "topic=true")
    assert cfg.member == "123" and cfg.topic == "true"
    with pytest.raises(ValueError, match="cannot parse"):
        ExperimentConfig().with_overrides("lr=abc")
    with pytest.raises(ValueError, match="epochs"):
        ExperimentConfig(epochs=0)


def test_mc_dropout_without_dropout_rejected():
    with pytest.raises(ValueError, match="dropout"):
        ExperimentConfig(mc_dropout_samples=4, dropout=0.0)


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"lrr": 1})
    with pytest.raises(ValueError):
        ExperimentConfig(model="vgg")


def test_json_file_is_flat_and_complete(tmp_path):
    ExperimentConfig().to_json(tmp_path / "c.json")
    d = json.loads((tmp_path / "c.json").read_text())
    assert {"model", "lr", "n_mels", "seed"} <= set(d)


def test_device_hip_alias():
    import torch

    from iml.train import GPU, get_device

    assert get_device("hip") == torch.device(GPU) == get_device("cuda")
    assert get_device("cpu").type == "cpu"
