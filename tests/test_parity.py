"""Numerical parity of the restructured paths against the reference forward."""

from __future__ import annotations

import threading

import pytest
import torch

from turbo_roformer.fast import FastBSRoformer

from .conftest import TINY_CHUNK, snr_db, synth_music


def _chunk(n=TINY_CHUNK, seed=0):
    return torch.from_numpy(synth_music(n, seed=seed)).unsqueeze(0)


@torch.inference_mode()
def test_fp32_restructured_matches_reference(tiny_model):
    x = _chunk()
    ref = tiny_model(x)
    out = FastBSRoformer(tiny_model, dtype=torch.float32, glue="torch", attn="sdpa", fused=())(x)
    assert out.shape == ref.shape
    assert snr_db(ref, out) >= 100.0


@torch.inference_mode()
@pytest.mark.parametrize("n", [TINY_CHUNK, 5000, 3 * 128 + 1])
def test_fp32_restructured_any_length(tiny_model, n):
    x = _chunk(n, seed=1)
    ref = tiny_model(x)
    out = FastBSRoformer(tiny_model, dtype=torch.float32, glue="torch", attn="sdpa", fused=())(x)
    assert out.shape == ref.shape
    assert snr_db(ref, out) >= 100.0


@torch.inference_mode()
def test_bf16_torch_path_close(tiny_model):
    x = _chunk()
    ref = tiny_model(x)
    out = FastBSRoformer(tiny_model, dtype=torch.bfloat16, glue="torch", attn="sdpa", fused=())(x)
    assert snr_db(ref, out) >= 30.0


KERNEL_CONFIGS = {
    "cpp_glue_sdpa": dict(glue="cpp", attn="sdpa", fused=()),
    "cpp_attention": dict(glue="cpp", attn="cpp", fused=()),
    "fused_best": dict(glue="cpp", attn="cpp", fused="best"),
    "fused_all": dict(glue="cpp", attn="cpp", fused="all"),
}


@pytest.mark.kernels
@pytest.mark.parametrize("cfg", list(KERNEL_CONFIGS))
def test_kernels_vs_fp32(tiny_model, kernels, cfg):
    x = _chunk()
    with torch.inference_mode():
        ref = FastBSRoformer(tiny_model, dtype=torch.float32, glue="torch", attn="sdpa", fused=())(x)
        out = FastBSRoformer(tiny_model, dtype=torch.bfloat16, **KERNEL_CONFIGS[cfg])(x)
    assert out.shape == ref.shape
    assert snr_db(ref, out) >= 40.0


@pytest.mark.kernels
def test_kernels_worker_thread(tiny_model, kernels):
    """Off the main thread the fused path switches to the allocation-free FF1 kernel - same result."""
    x = _chunk(seed=3)
    fast = FastBSRoformer(tiny_model, dtype=torch.bfloat16, glue="cpp", attn="cpp", fused="best")
    with torch.inference_mode():
        ref = FastBSRoformer(tiny_model, dtype=torch.float32, glue="torch", attn="sdpa", fused=())(x)
        main = fast(x)
    res = {}

    def work():
        with torch.inference_mode():
            res["y"] = fast(x)

    th = threading.Thread(target=work)
    th.start()
    th.join()
    assert snr_db(ref, res["y"]) >= 40.0
    assert snr_db(main, res["y"]) >= 40.0


@pytest.mark.kernels
@pytest.mark.parametrize("n", [5000, 3 * 128 + 1, 2 * TINY_CHUNK + 777])
def test_kernels_odd_lengths(tiny_model, kernels, n):
    x = _chunk(n, seed=2)
    with torch.inference_mode():
        ref = FastBSRoformer(tiny_model, dtype=torch.float32, glue="torch", attn="sdpa", fused=())(x)
        out = FastBSRoformer(tiny_model, dtype=torch.bfloat16, glue="cpp", attn="cpp", fused="best")(x)
    assert snr_db(ref, out) >= 40.0


# ---------------------------------------------------------------- real weights


@pytest.mark.weights
@pytest.mark.slow
def test_real_model_fp32_restructured_matches_reference(real_separator_fp32):
    ref_model = real_separator_fp32.net.model
    x = torch.from_numpy(synth_music(3 * 44100, seed=5)).unsqueeze(0)
    with torch.inference_mode():
        ref = ref_model(x)
        out = real_separator_fp32.net(x)
    assert snr_db(ref, out) >= 100.0


@pytest.mark.weights
@pytest.mark.kernels
@pytest.mark.slow
def test_real_model_kernels_vs_fp32_full_chunk(real_separator_fp32, kernels):
    sep = real_separator_fp32
    x = torch.from_numpy(synth_music(sep.chunk, seed=6)).unsqueeze(0)
    fast = FastBSRoformer(sep.net.model, dtype=torch.bfloat16, glue="cpp", attn="cpp", fused="best")
    with torch.inference_mode():
        ref = sep.net(x)
        out = fast(x)
    assert snr_db(ref, out) >= 40.0


@pytest.mark.weights
@pytest.mark.slow
def test_reference_matches_audio_separator(real_separator_fp32):
    """Our einops-free reference module vs python-audio-separator's BSRoformer (optional dependency)."""
    bs = pytest.importorskip("audio_separator.separator.uvr_lib_v5.roformer.bs_roformer")
    from turbo_roformer import registry

    spec = registry.MODELS["resurrection"]
    cfg = dict(spec.model_config)
    dim = cfg.pop("dim")
    upstream = bs.BSRoformer(dim, **cfg)
    upstream.load_state_dict(real_separator_fp32.net.model.state_dict(), strict=True)
    upstream.eval()
    x = torch.from_numpy(synth_music(2 * 44100, seed=7)).unsqueeze(0)
    with torch.inference_mode():
        ours = real_separator_fp32.net.model(x)
        theirs = upstream(x)
    assert snr_db(theirs, ours) >= 100.0
