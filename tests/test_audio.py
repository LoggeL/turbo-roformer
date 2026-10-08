"""Audio helpers: formats, channel layouts, resampling."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from turbo_roformer import audio


@pytest.mark.parametrize("fmt", audio.output_formats())
def test_write_and_read_back(tmp_path, fmt):
    x = (0.1 * np.random.default_rng(0).standard_normal((2, 44100))).astype(np.float32)
    p = tmp_path / f"x.{fmt}"
    audio.write(p, x, 44100, fmt)
    y, sr = audio.load(p)
    assert sr == 44100 and y.shape[0] == 2 and abs(y.shape[1] - 44100) < 3000
    if fmt == "wav":
        np.testing.assert_array_equal(x, y)


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        audio.write(tmp_path / "x.aiff", np.zeros((2, 10), np.float32), 44100, "aiff")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_fallback(tmp_path):
    src = tmp_path / "x.wav"
    audio.write(src, np.zeros((2, 4410), np.float32), 44100, "wav")
    m4a = tmp_path / "x.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "aac", str(m4a)], check=True)
    y, sr = audio.load(m4a)
    assert sr == 44100 and y.shape[0] == 2


def test_resample_length_and_identity():
    x = np.random.default_rng(1).standard_normal((2, 48000)).astype(np.float32)
    y = audio.resample(x, 48000, 44100, length=44100)
    assert y.shape == (2, 44100) and y.dtype == np.float32
    np.testing.assert_array_equal(audio.resample(x, 48000, 48000), x)


def test_downmix_gains_are_normalised():
    ones = np.ones((6, 10), np.float32)
    np.testing.assert_allclose(audio.downmix_stereo(ones), 1.0, rtol=1e-6)
    three = np.ones((3, 10), np.float32)
    np.testing.assert_allclose(audio.downmix_stereo(three), 1.0, rtol=1e-6)


def test_as_channels_first():
    assert audio.as_channels_first(np.zeros(5, np.float64)).shape == (1, 5)
    assert audio.as_channels_first(np.zeros(5, np.float64)).dtype == np.float32
    with pytest.raises(ValueError):
        audio.as_channels_first(np.zeros((20, 3), np.float32))
