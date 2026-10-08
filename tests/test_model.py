"""Reference module: checkpoint compatibility (state-dict layout) and basic behaviour."""

from __future__ import annotations

import pytest
import torch

from turbo_roformer.model import BSRoformer, build_model
from turbo_roformer.registry import MODELS

# a few parameter names exactly as they appear in audio-separator / lucidrains checkpoints
UPSTREAM_KEYS = [
    "layers.0.0.layers.0.0.rotary_embed.freqs",
    "layers.0.0.layers.0.0.norm.gamma",
    "layers.0.0.layers.0.0.to_qkv.weight",
    "layers.0.0.layers.0.0.to_gates.weight",
    "layers.0.0.layers.0.0.to_gates.bias",
    "layers.0.0.layers.0.0.to_out.0.weight",
    "layers.0.1.layers.0.1.net.0.gamma",
    "layers.0.1.layers.0.1.net.1.weight",
    "layers.0.1.layers.0.1.net.4.bias",
    "final_norm.gamma",
    "band_split.to_features.0.0.gamma",
    "band_split.to_features.61.1.weight",
    "mask_estimators.0.to_freqs.0.0.0.weight",
    "mask_estimators.0.to_freqs.61.0.2.bias",
]


@pytest.fixture(scope="module")
def full_arch():
    return build_model(MODELS["resurrection"].model_config)


def test_state_dict_layout(full_arch):
    keys = set(full_arch.state_dict())
    for k in UPSTREAM_KEYS:
        assert k in keys, k
    n_params = sum(p.numel() for p in full_arch.parameters())
    # rotary freqs are shared buffers-as-parameters; ~51 M weights like the 204 MB fp32 checkpoint
    assert 50e6 < n_params < 52e6


def test_forward_shape(tiny_model):
    x = torch.randn(1, 2, 5000)
    with torch.inference_mode():
        y = tiny_model(x)
    assert y.shape == (1, 2, 128 * (5000 // 128))


def test_rejects_linear_transformer():
    with pytest.raises(NotImplementedError):
        BSRoformer(32, depth=1, linear_transformer_depth=1, stft_n_fft=64, freqs_per_bands=(33,), stereo=True)


@pytest.mark.weights
def test_real_checkpoint_loads_strict(real_separator_fp32):
    m = real_separator_fp32.net.model
    assert sum(p.numel() for p in m.parameters()) > 50e6
