import pytest
import torch

from iml.config import MODEL_NAMES, ExperimentConfig
from iml.models import SUPPORTED_KNOBS, build_model, count_params


def small(name, **kw):
    """Config with a small width where the model supports it (speed), defaults otherwise."""
    if "width" in SUPPORTED_KNOBS[name]:
        kw.setdefault("width", 8)
    return ExperimentConfig(model=name, **kw)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("T", [101, 201])  # 1 s and 2 s at hop 10 ms
def test_forward_shape_any_length(name, T):
    cfg = small(name, n_mels=64)
    model = build_model(cfg).eval()
    x = torch.randn(2, 1, cfg.n_mels, T)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2)
    assert torch.isfinite(y).all()
    assert count_params(model) > 0


@pytest.mark.parametrize("bn", ["before", "after", "none"])
@pytest.mark.parametrize("act", ["relu", "sigmoid"])
@pytest.mark.parametrize("init", ["he", "xavier", "uniform", "default"])
def test_simple_cnn_knobs(bn, act, init):
    cfg = ExperimentConfig(model="simple_cnn", width=4, bn_position=bn, activation=act, init=init)
    y = build_model(cfg)(torch.randn(3, 1, 64, 50))
    assert y.shape == (3, 2)


def test_cnn10_depth_changes_params():
    small = count_params(build_model(ExperimentConfig(model="cnn10", width=8, depth=2)))
    big = count_params(build_model(ExperimentConfig(model="cnn10", width=8, depth=4)))
    assert big > small


def test_ecapa_embedding():
    model = build_model(small("ecapa_tdnn")).eval()
    with torch.no_grad():
        emb = model.embed(torch.randn(2, 1, 64, 120))
    assert emb.shape == (2, 192)


def test_models_train_step_backprop():
    for name in MODEL_NAMES:
        model = build_model(small(name)).train()
        x = torch.randn(4, 1, 64, 80)
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads), name


@pytest.mark.parametrize(
    "name, knob",
    [
        ("resnet18", {"width": 8}),
        ("mobilenet_v2", {"bn_position": "after"}),
        ("ecapa_tdnn", {"depth": 3}),
    ],
)
def test_ignored_knob_is_rejected(name, knob):
    with pytest.raises(ValueError, match="ignores"):
        build_model(ExperimentConfig(model=name, **knob))


def test_init_rejected_with_pretrained_weights():
    with pytest.raises(ValueError, match="ignores"):
        build_model(ExperimentConfig(model="resnet18", pretrained=True, init="xavier"))


def test_conv_bias_only_dropped_before_bn():
    from iml.models.common import conv_block

    assert conv_block(1, 4, 3, "before", "relu")[0].bias is None
    assert conv_block(1, 4, 3, "after", "relu")[0].bias is not None
    assert conv_block(1, 4, 3, "none", "relu")[0].bias is not None
