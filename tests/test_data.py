import torch

from iml.cli._data import load_splits
from iml.config import ExperimentConfig
from iml.data.audio import crop_or_pad, load_wav, save_wav
from iml.data.dataset import SpectrogramDataset
from iml.data.spectrogram import LogMelSpectrogram, n_frames


def test_logmel_shape_for_1s_and_3s():
    cfg = ExperimentConfig()
    spec = LogMelSpectrogram(cfg)
    for sec in (1.0, 3.0):
        wave = torch.randn(int(sec * cfg.sample_rate))
        out = spec(wave)
        assert out.shape == (1, 1, cfg.n_mels, int(sec * cfg.sample_rate) // cfg.hop_length + 1)
    assert abs(out.mean()) < 1e-3 and abs(out.std() - 1) < 1e-2  # per_sample normalisation


def test_crop_or_pad():
    w = torch.arange(10.0)
    assert crop_or_pad(w, 4).tolist() == [3.0, 4.0, 5.0, 6.0]
    assert crop_or_pad(w, 12).shape == (12,) and crop_or_pad(w, 12)[-2:].tolist() == [0.0, 0.0]
    assert crop_or_pad(w, 4, random_crop=True).shape == (4,)


def test_wav_roundtrip_and_resample(tmp_path):
    cfg = ExperimentConfig(sample_rate=16000)
    save_wav(tmp_path / "a.wav", torch.randn(44100) * 0.1, 44100)
    wave = load_wav(tmp_path / "a.wav", cfg.sample_rate)
    assert abs(wave.shape[0] - 16000) <= 1
    ds = SpectrogramDataset([(tmp_path / "a.wav", 1)], cfg, train=False)
    spec, label = ds[0]
    assert spec.shape == (1, cfg.n_mels, n_frames(cfg)) and label == 1


def test_contract_dataset_loaders(tiny_data):
    cfg = ExperimentConfig(batch_size=8, num_workers=0)
    loaders = load_splits(cfg, data_dir=tiny_data)
    assert set(loaders) == {"train", "val", "test"}
    x, y = loaders["val"].dataset[0]
    assert x.shape == (1, cfg.n_mels, n_frames(cfg)) and y in (0, 1)
    xb, yb = next(iter(loaders["train"]))
    assert xb.shape[0] == 8 and yb.shape == (8,)
