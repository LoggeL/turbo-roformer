"""Shared helpers for the benchmark scripts (synthetic audio, metrics, peak memory)."""

from __future__ import annotations

import resource
import sys

import numpy as np


def synthetic_mix(seconds: float, sr: int = 44100, seed: int = 0) -> np.ndarray:
    """Deterministic, music-like stereo signal (float32, (2, n)).

    Harmonic notes with decays, a vibrato 'voice' line panned centre, noise bursts. Runtime of the model does not
    depend on the content, so this is all a speed benchmark needs; quality is measured model-vs-model on it.
    """
    rng = np.random.default_rng(seed)
    n = round(seconds * sr)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros((2, n))
    note_len = int(0.5 * sr)
    for start in range(0, n, note_len // 2):
        f0 = 110.0 * 2 ** (rng.integers(0, 24) / 12)
        seg = slice(start, min(n, start + 2 * note_len))
        tt = t[seg] - t[start]
        env = np.exp(-tt * rng.uniform(1.5, 5.0))
        tone = sum(np.sin(2 * np.pi * f0 * h * tt) / h for h in range(1, 7)) * env
        pan = rng.uniform(0.2, 0.8)
        out[0, seg] += (1 - pan) * tone
        out[1, seg] += pan * tone
    voice = 0.4 * np.sin(2 * np.pi * 220 * t + 3 * np.sin(2 * np.pi * 5.5 * t)) * (np.sin(2 * np.pi * 0.25 * t) > 0)
    out += voice
    out += 0.03 * rng.standard_normal((2, n))
    out *= 0.8 / np.abs(out).max()
    return out.astype(np.float32)


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref = np.asarray(ref, np.float64)
    est = np.asarray(est, np.float64)
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n], est[..., :n]
    return float(10 * np.log10((ref**2).sum() / max(((ref - est) ** 2).sum(), 1e-300)))


def windowed_snr_db(ref: np.ndarray, est: np.ndarray, sr: int = 44100, win_s: float = 1.0, floor_dbfs: float = -50):
    """(worst, median) SNR over 1 s windows whose reference RMS is above ``floor_dbfs``."""
    ref = np.asarray(ref, np.float64)
    est = np.asarray(est, np.float64)
    w = int(win_s * sr)
    nw = min(ref.shape[-1], est.shape[-1]) // w
    vals = []
    for i in range(nw):
        r = ref[..., i * w : (i + 1) * w]
        e = est[..., i * w : (i + 1) * w] - r
        if 10 * np.log10((r**2).mean() + 1e-30) > floor_dbfs:
            vals.append(10 * np.log10((r**2).sum() / max((e**2).sum(), 1e-300)))
    return (float(min(vals)), float(np.median(vals))) if vals else (float("nan"), float("nan"))


def peak_rss_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024**3 if sys.platform == "darwin" else 1024**2)
