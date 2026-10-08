"""Separator API: shapes, channel layouts, sample rates, edge cases, error handling."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from turbo_roformer import Separator
from turbo_roformer.audio import downmix_stereo

from .conftest import TINY_CHUNK, TINY_DIM_T, TINY_HOP, snr_db, synth_music

SR = 44100


@pytest.fixture(scope="module")
def sep(tiny_model):
    return Separator.from_module(tiny_model, TINY_HOP, TINY_DIM_T, precision="fp32", backend="torch")


@pytest.fixture(scope="module")
def sep_kernels(tiny_model, kernels):
    return Separator.from_module(tiny_model, TINY_HOP, TINY_DIM_T, precision="bf16", backend="kernels")


def test_stereo_shape_and_dtype(sep):
    mix = synth_music(3 * SR)
    inst = sep.separate(mix)
    assert inst.shape == mix.shape and inst.dtype == np.float32
    assert np.isfinite(inst).all()
    assert sep.n_chunks(mix.shape[1], 2.0) > 2  # 3 s spans many tiny-model chunks


def test_deterministic(sep):
    mix = synth_music(SR, seed=4)
    np.testing.assert_array_equal(sep.separate(mix), sep.separate(mix))


def test_mono_1d_and_2d(sep):
    mono = synth_music(SR, channels=1)
    a = sep.separate(mono[0])
    b = sep.separate(mono)
    c = sep.separate(np.repeat(mono, 2, axis=0))
    assert a.shape == (2, SR)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_surround_51_downmix_keeps_centre(sep):
    six = synth_music(SR, channels=6, seed=2)
    inst = sep.separate(six)
    assert inst.shape == (2, SR)
    np.testing.assert_array_equal(inst, sep.separate(downmix_stereo(six)))
    centre_only = np.zeros_like(six)
    centre_only[2] = six[2]  # vocals usually live in the centre channel: it must survive the downmix
    dm = downmix_stereo(centre_only)
    assert np.abs(dm).max() > 0.2 * np.abs(six[2]).max()
    np.testing.assert_allclose(dm[0], dm[1])


@pytest.mark.parametrize("n", [1, 2, 127, 128, 383, 385, TINY_CHUNK - 1, TINY_CHUNK, TINY_CHUNK + 1])
def test_short_and_boundary_lengths(sep, n):
    mix = synth_music(max(n, 2), seed=n)[:, :n]
    inst = sep.separate(mix)
    assert inst.shape == (2, n)
    assert np.isfinite(inst).all()


def test_three_seconds_vocals_identity(sep):
    mix = synth_music(3 * SR, seed=9)
    inst = sep.separate(mix)
    vocals = mix - inst
    np.testing.assert_allclose(inst + vocals, mix, atol=1e-6)


@pytest.mark.parametrize("sr", [48000, 22050, 32000])
def test_other_sample_rates(sep, sr):
    n = int(1.5 * sr)
    mix = synth_music(n, sr=sr, seed=sr)
    inst = sep.separate(mix, sr=sr)
    assert inst.shape == (2, n) and inst.dtype == np.float32
    assert np.isfinite(inst).all()


def test_48k_close_to_44k(sep):
    """Resampling round trip: separating a 48 kHz copy ~ separating the 44.1 kHz original."""
    from turbo_roformer.audio import resample

    mix44 = synth_music(2 * SR, seed=11)
    mix48 = resample(mix44, SR, 48000)
    inst44 = sep.separate(mix44)
    inst48 = resample(sep.separate(mix48, sr=48000), 48000, SR, length=mix44.shape[1])
    assert snr_db(inst44[:, 2000:-2000], inst48[:, 2000:-2000]) > 20


def test_peak_normalisation_is_scale_invariant(sep):
    mix = synth_music(SR, seed=12)
    a = sep.separate(mix * 2.0) / 2.0
    b = sep.separate(mix * 4.0) / 4.0
    assert snr_db(a, b) > 100


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.0, np.nan, 0.0]] * 2, np.float32),
        np.array([[0.0, np.inf, 0.0]] * 2, np.float32),
        np.zeros((2, 0), np.float32),
        np.zeros((2, 10), np.int16),
        np.zeros((1000, 2), np.float32),  # (samples, channels)
        np.zeros((2, 2, 10), np.float32),
    ],
)
def test_invalid_input_raises(sep, bad):
    with pytest.raises(ValueError):
        sep.separate(bad)


def test_invalid_overlap_and_sr(sep):
    mix = synth_music(1000)
    with pytest.raises(ValueError):
        sep.separate(mix, overlap=0.5)
    with pytest.raises(ValueError):
        sep.separate(mix, sr=0)


def test_progress_callback(sep):
    mix = synth_music(SR)
    calls = []
    sep.separate(mix, progress=lambda k, n: calls.append((k, n)))
    n = sep.n_chunks(SR)
    assert calls == [(k, n) for k in range(1, n + 1)]


@pytest.mark.parametrize("overlap", [1.0, 1.5, 3.0])
def test_overlaps(sep, overlap):
    mix = synth_music(SR, seed=13)
    assert sep.separate(mix, overlap=overlap).shape == mix.shape


def test_concurrent_calls_are_serialised(sep):
    mix = synth_music(SR, seed=14)
    expected = sep.separate(mix)
    results = [None] * 3

    def work(i):
        results[i] = sep.separate(mix)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        np.testing.assert_array_equal(r, expected)


def test_separate_file_roundtrip(sep, tmp_path):
    import soundfile as sf

    mix = synth_music(SR, sr=48000, seed=15)
    src = tmp_path / "in.wav"
    sf.write(src, mix.T, 48000, subtype="FLOAT")
    res = sep.separate_file(src, tmp_path / "out", formats=("wav", "flac"))
    inst, sr = sf.read(res["instrumental"], dtype="float32", always_2d=True)
    voc, _ = sf.read(res["vocals"], dtype="float32", always_2d=True)
    assert sr == 48000 and inst.shape == mix.T.shape
    np.testing.assert_allclose(inst + voc, mix.T, atol=1e-6)
    assert set(res["files"]) == {"instrumental.wav", "vocals.wav", "instrumental.flac", "vocals.flac"}
    assert res["rtf"] > 0 and res["duration_s"] == pytest.approx(SR / 48000)


@pytest.mark.kernels
def test_kernels_backend_close_to_fp32(sep, sep_kernels):
    mix = synth_music(3 * SR, seed=16)
    assert sep_kernels.backend == "kernels"
    assert snr_db(sep.separate(mix), sep_kernels.separate(mix)) >= 40.0


@pytest.mark.kernels
def test_kernels_backend_edge_cases(sep_kernels):
    for n in (1, 385, TINY_CHUNK + 1):
        assert sep_kernels.separate(synth_music(max(n, 2))[:, :n]).shape == (2, n)
    assert sep_kernels.separate(synth_music(SR, sr=48000), sr=48000).shape == (2, SR)
