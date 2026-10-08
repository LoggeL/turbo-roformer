"""Compare two separations of the same song (e.g. turbo-roformer vs stock audio-separator).

    python benchmarks/compare_wavs.py reference_instrumental.wav candidate_instrumental.wav

Prints SNR, scale-invariant SNR and the worst/median 1-s window SNR (windows above -50 dBFS RMS).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf

from _common import snr_db, windowed_snr_db


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference")
    ap.add_argument("candidate")
    a = ap.parse_args()
    ref, sr1 = sf.read(a.reference, dtype="float64", always_2d=True)
    est, sr2 = sf.read(a.candidate, dtype="float64", always_2d=True)
    if sr1 != sr2:
        raise SystemExit(f"sample rates differ: {sr1} vs {sr2}")
    n = min(len(ref), len(est))
    ref, est = ref[:n].T, est[:n].T
    g = float((est * ref).sum() / max((est * est).sum(), 1e-300))
    worst, median = windowed_snr_db(ref, est, sr1)
    res = dict(
        samples=n,
        identical=bool(np.array_equal(ref, est)),
        snr_db=round(snr_db(ref, est), 2),
        si_snr_db=round(snr_db(ref, g * est), 2),
        worst_1s_snr_db=round(worst, 2),
        median_1s_snr_db=round(median, 2),
        max_abs_diff=float(np.abs(ref - est).max()),
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
