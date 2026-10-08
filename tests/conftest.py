"""Shared fixtures: a small random BS-RoFormer (no download needed) and opt-in real weights."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from turbo_roformer import _kernels
from turbo_roformer.model import BSRoformer

# small but structurally complete: dim 256 / 8 heads / dim_head 64 (what the fused kernels need), 2 axial blocks,
# n_fft 512 (257 bins in 29 bands), hop 128, 128-frame chunks (16256 samples, ~0.37 s)
TINY_CFG = dict(
    dim=256,
    depth=2,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    freqs_per_bands=(4,) * 16 + (8,) * 8 + (16,) * 4 + (65,),
    dim_head=64,
    heads=8,
    dim_freqs_in=257,
    stft_n_fft=512,
    stft_hop_length=128,
    stft_win_length=512,
    mask_estimator_depth=2,
    mlp_expansion_factor=2,
)
TINY_HOP = 128
TINY_DIM_T = 128
TINY_CHUNK = TINY_HOP * (TINY_DIM_T - 1)


def pytest_addoption(parser):
    parser.addoption("--weights", action="store_true", help="run tests that need the real model weights (~200 MB)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--weights"):
        return
    skip = pytest.mark.skip(reason="needs real weights; run with --weights")
    for item in items:
        if "weights" in item.keywords:
            item.add_marker(skip)


def make_tiny_model(seed: int = 0) -> BSRoformer:
    torch.manual_seed(seed)
    m = BSRoformer(TINY_CFG["dim"], **{k: v for k, v in TINY_CFG.items() if k != "dim"})
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("gamma"):  # non-trivial norm gains so the folding is actually exercised
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            elif name.endswith("bias"):
                p.copy_(0.05 * torch.randn_like(p))
    return m.eval()


@pytest.fixture(scope="session")
def tiny_model() -> BSRoformer:
    return make_tiny_model()


@pytest.fixture(scope="session")
def kernels():
    ok, reason = _kernels.support()
    if not ok:
        pytest.skip(f"AVX512-BF16 kernels unavailable: {reason}")
    try:
        _kernels.load()
    except _kernels.KernelUnavailable as e:
        pytest.skip(str(e))
    return torch.ops.turbo_roformer


def synth_music(n: int, sr: int = 44100, channels: int = 2, seed: int = 0) -> np.ndarray:
    """Deterministic music-like test signal: decaying harmonic notes, a vibrato 'voice', noise hits."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    out = np.zeros((channels, n))
    for c in range(channels):
        sig = np.zeros(n)
        for _ in range(6):
            f0 = 110.0 * 2 ** (rng.integers(0, 24) / 12)
            start = rng.uniform(0, max(t[-1], 1e-3))
            env = np.where(t >= start, np.exp(-(t - start) * rng.uniform(1, 4)), 0.0)
            sig += env * sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3)) / h for h in range(1, 6))
        sig += 0.3 * np.sin(2 * np.pi * 330 * t + 4 * np.sin(2 * np.pi * 5 * t))
        sig += 0.05 * rng.standard_normal(n) * (np.sin(2 * np.pi * 2 * t) > 0.9)
        out[c] = sig
    out *= 0.5 / max(np.abs(out).max(), 1e-9)
    return out.astype(np.float32)


def snr_db(ref, est) -> float:
    ref = np.asarray(ref, np.float64)
    est = np.asarray(est, np.float64)
    return float(10 * np.log10((ref**2).sum() / max(((ref - est) ** 2).sum(), 1e-300)))


@pytest.fixture(scope="session")
def real_separator_fp32():
    from turbo_roformer import Separator

    return Separator("resurrection", precision="fp32", backend="torch", threads=None)
