"""Reproducible speed benchmark on a synthetic song (no audio files needed).

    python benchmarks/bench_synthetic.py                     # 4:12 synthetic song, auto backend, 8 threads
    python benchmarks/bench_synthetic.py --seconds 60 --worker-thread
    python benchmarks/bench_synthetic.py --quality-seconds 34   # + bf16 vs exact fp32 SNR on the first 34 s

Prints load time, per-chunk times, wall time, RTF (= wall / audio duration, model load excluded) and peak RSS.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import threading
import time

from _common import peak_rss_gb, snr_db, synthetic_mix
from turbo_roformer import Separator, _kernels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=252.39, help="synthetic song length (default 4:12)")
    ap.add_argument("--overlap", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--backend", default="auto", choices=["auto", "kernels", "torch"])
    ap.add_argument("--worker-thread", action="store_true", help="call separate() from a non-main thread")
    ap.add_argument(
        "--quality-seconds",
        type=float,
        default=0.0,
        help="also separate the first N seconds with the exact fp32 path and report the SNR",
    )
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    mix = synthetic_mix(a.seconds)
    sep = Separator(precision=a.precision, threads=a.threads, backend=a.backend)
    times: list[float] = []
    last = [time.perf_counter()]

    def progress(k: int, n: int) -> None:
        now = time.perf_counter()
        times.append(now - last[0])
        last[0] = now

    out: dict = {}

    def run() -> None:
        t0 = time.perf_counter()
        last[0] = t0
        out["inst"] = sep.separate(mix, overlap=a.overlap, progress=progress)
        out["wall"] = time.perf_counter() - t0

    if a.worker_thread:
        th = threading.Thread(target=run)
        th.start()
        th.join()
    else:
        run()

    res = dict(
        cpu=_kernels.cpu_model_name(),
        python=platform.python_version(),
        backend=sep.backend,
        precision=sep.precision,
        threads=a.threads,
        worker_thread=a.worker_thread,
        overlap=a.overlap,
        audio_s=round(a.seconds, 2),
        load_s=round(sep.load_s, 2),
        n_chunks=len(times),
        chunk_s_median=round(statistics.median(times), 3),
        chunk_s_min=round(min(times), 3),
        wall_s=round(out["wall"], 2),
        rtf=round(out["wall"] / a.seconds, 4),
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    if a.quality_seconds > 0:
        n = int(a.quality_seconds * 44100)
        ref = Separator(precision="fp32", threads=a.threads, backend="torch")
        exact = ref.separate(mix[:, :n], overlap=a.overlap)
        fast = sep.separate(mix[:, :n], overlap=a.overlap)
        res["snr_vs_fp32_db"] = round(snr_db(exact, fast), 2)
    if a.json:
        print(json.dumps(res))
    else:
        for k, v in res.items():
            print(f"{k:>16}: {v}")


if __name__ == "__main__":
    main()
