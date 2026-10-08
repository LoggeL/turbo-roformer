"""How much error does lossy encoding add compared with bf16-vs-fp32 differences?

    python benchmarks/codec_floor.py instrumental.wav --bitrates 128k 192k 320k

Encodes the file with ffmpeg (libmp3lame), decodes it, aligns it (small lag search) and prints
SNR(decoded, original). If this number is far below the bf16-vs-fp32 SNR, the precision difference is
irrelevant for an MP3 deliverable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from _common import snr_db


def roundtrip(path: str, bitrate: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        mp3, dec = Path(td) / "x.mp3", Path(td) / "x.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", path, "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)], check=True
        )
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-c:a", "pcm_f32le", str(dec)], check=True)
        y, _ = sf.read(dec, dtype="float64", always_2d=True)
    return y.T


def align(ref: np.ndarray, y: np.ndarray, max_lag: int = 3000) -> np.ndarray:
    seg = slice(44100, 44100 * 11)  # 10 s probe
    r = ref[:, seg].mean(0)
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1, 1):
        s = y[:, seg.start + lag : seg.stop + lag].mean(0)
        if len(s) != len(r):
            continue
        c = float(np.dot(r, s))
        if c > best:
            best, best_lag = c, lag
    return y[:, best_lag:] if best_lag >= 0 else np.pad(y, ((0, 0), (-best_lag, 0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wav")
    ap.add_argument("--bitrates", nargs="+", default=["128k", "192k", "320k"])
    a = ap.parse_args()
    x, _ = sf.read(a.wav, dtype="float64", always_2d=True)
    x = x.T
    for br in a.bitrates:
        y = align(x, roundtrip(a.wav, br))
        print(json.dumps({"bitrate": br, "snr_mp3_vs_original_db": round(snr_db(x, y), 2)}))


if __name__ == "__main__":
    main()
