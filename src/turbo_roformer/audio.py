"""Audio helpers: channel handling, resampling, file I/O."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

MAX_CHANNELS = 16


def as_channels_first(mix) -> np.ndarray:
    """Validate and convert to a float32 array of shape (channels, samples)."""
    a = np.asarray(mix)
    if a.dtype.kind not in "f":
        raise ValueError(f"expected a floating point array (got dtype {a.dtype}); scale integer PCM to [-1, 1] first")
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim != 2:
        raise ValueError(f"expected (channels, samples) or (samples,), got shape {a.shape}")
    if a.shape[0] > MAX_CHANNELS:
        raise ValueError(f"got {a.shape[0]} channels - is the array (samples, channels)? pass mix.T")
    if a.shape[1] == 0 or a.shape[0] == 0:
        raise ValueError("empty input")
    a = a.astype(np.float32, copy=False)
    if not np.isfinite(a).all():
        raise ValueError("input contains NaN or inf samples")
    return a


def downmix_stereo(x: np.ndarray) -> np.ndarray:
    """(C>2, n) -> (2, n), assuming WAV/FFmpeg channel order FL FR FC [LFE] surrounds...

    The centre goes into both sides at -3 dB (it usually carries the vocals, so it must not be dropped),
    LFE is dropped for C >= 6, the remaining surround channels alternate left/right at -3 dB, and each side is
    normalised by its summed gains.
    """
    c = x.shape[0]
    a = 0.7071
    L, R = x[0].astype(np.float64), x[1].astype(np.float64)
    gl = gr = 1.0
    if c >= 3:
        L = L + a * x[2]
        R = R + a * x[2]
        gl += a
        gr += a
    rest = list(range(4 if c >= 6 else 3, c))
    for k, ch in enumerate(rest):
        if k % 2 == 0:
            L = L + a * x[ch]
            gl += a
        else:
            R = R + a * x[ch]
            gr += a
    return np.stack([L / gl, R / gr]).astype(np.float32)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """(C, n) float32 -> (2, n): mono is duplicated, stereo passes through, more channels are downmixed."""
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    if x.shape[0] == 2:
        return np.ascontiguousarray(x)
    return downmix_stereo(x)


def resample(x: np.ndarray, sr_in: int, sr_out: int, length: int | None = None) -> np.ndarray:
    """(C, n) -> (C, m) with soxr (HQ). ``length`` forces the output length (zero-pad / trim)."""
    if sr_in != sr_out:
        import soxr

        y = soxr.resample(np.ascontiguousarray(x.T), sr_in, sr_out, quality="HQ").T
        y = np.asarray(y, dtype=np.float32).reshape(x.shape[0], -1)
    else:
        y = x
    if length is not None and y.shape[1] != length:
        y = y[:, :length] if y.shape[1] > length else np.pad(y, ((0, 0), (0, length - y.shape[1])))
    return np.ascontiguousarray(y, dtype=np.float32)


def load(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Read an audio file -> ((channels, samples) float32, sample_rate).

    Uses libsndfile (WAV, FLAC, OGG, MP3 with libsndfile >= 1.1) and falls back to ``ffmpeg`` on PATH for
    anything else (AAC/M4A, ...).
    """
    import soundfile as sf

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(y.T), int(sr)
    except Exception as e:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"cannot decode {path}: {e} (install ffmpeg for more formats)") from e
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "decoded.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(wav)], check=True
        )
        y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    return np.ascontiguousarray(y.T), int(sr)


_FORMATS = {
    "wav": dict(format="WAV", subtype="FLOAT"),
    "flac": dict(format="FLAC", subtype="PCM_24"),
    "ogg": dict(format="OGG", subtype="VORBIS"),
    "mp3": dict(format="MP3", subtype="MPEG_LAYER_III"),
}


def write(path: str | os.PathLike, x: np.ndarray, sr: int, fmt: str) -> None:
    """Write (channels, samples) float32. WAV keeps float32; other formats are clipped to [-1, 1]."""
    import soundfile as sf

    if fmt not in _FORMATS:
        raise ValueError(f"unsupported output format {fmt!r} (choose from {', '.join(_FORMATS)})")
    data = np.ascontiguousarray(x.T, dtype=np.float32)
    if fmt != "wav":
        data = np.clip(data, -1.0, 1.0)
    sf.write(str(path), data, sr, **_FORMATS[fmt])


def output_formats() -> tuple[str, ...]:
    return tuple(_FORMATS)
