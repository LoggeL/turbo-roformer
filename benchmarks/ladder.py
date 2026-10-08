"""Per-chunk timing ladder: how each optimisation step changes the time for one 17 s model chunk.

    python benchmarks/ladder.py                 # all stages, 8 threads, prints a markdown table
    python benchmarks/ladder.py --stages best best_worker --repeats 3

Every stage runs in a fresh subprocess (the malloc tuning is process-wide and irreversible), on the same
synthetic 749259-sample chunk (runtime does not depend on the content). SNR is measured against the
"stock" stage, i.e. the unmodified reference forward in fp32.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

STAGES = {
    # name: (description, kind, options)
    "stock": ("reference forward, fp32 (= audio-separator's model code)", "ref", {}),
    "stock_malloc": ("+ glibc malloc tuning (no mmap churn)", "ref", {"malloc": True}),
    "stock_autocast": ("+ torch.autocast bf16", "ref", {"malloc": True, "autocast": True}),
    "restructured_fp32": (
        "restructured forward, fp32 (exact)",
        "fast",
        {"dtype": "fp32", "glue": "torch", "attn": "sdpa", "fused": ()},
    ),
    "restructured_bf16": (
        "restructured forward, bf16 GEMMs + bf16 SDPA",
        "fast",
        {"dtype": "bf16", "glue": "torch", "attn": "sdpa", "fused": ()},
    ),
    "cpp_glue_sdpa": (
        "+ C++ norm/rotary/packing glue (torch bf16 SDPA)",
        "fast",
        {"dtype": "bf16", "glue": "cpp", "attn": "sdpa", "fused": ()},
    ),
    "cpp_attention": (
        "+ custom AVX512-BF16 attention kernel",
        "fast",
        {"dtype": "bf16", "glue": "cpp", "attn": "cpp", "fused": ()},
    ),
    "fused_qkv": (
        "+ fused qkv GEMM with rotary/packing epilogue",
        "fast",
        {"dtype": "bf16", "glue": "cpp", "attn": "cpp", "fused": ("qkv",)},
    ),
    "fused_out": (
        "+ fused out-proj + residual + RMSNorm",
        "fast",
        {"dtype": "bf16", "glue": "cpp", "attn": "cpp", "fused": ("qkv", "out")},
    ),
    "best": (
        "+ fused FF2 + residual + RMSNorm (default)",
        "fast",
        {"dtype": "bf16", "glue": "cpp", "attn": "cpp", "fused": ("qkv", "out", "ff2")},
    ),
    "best_worker": (
        "default, called from a worker thread",
        "fast",
        {"dtype": "bf16", "glue": "cpp", "attn": "cpp", "fused": ("qkv", "out", "ff2"), "worker": True},
    ),
}


def run_stage(name: str, threads: int, repeats: int, out_dir: Path) -> dict:
    import torch

    from turbo_roformer import registry
    from turbo_roformer.fast import FastBSRoformer
    from turbo_roformer.model import build_model, load_checkpoint
    from turbo_roformer.separator import tune_malloc

    sys.path.insert(0, str(Path(__file__).parent))
    from _common import synthetic_mix

    desc, kind, opt = STAGES[name]
    torch.set_num_threads(threads)
    if opt.get("malloc") or kind == "fast":
        tune_malloc()
    spec = registry.MODELS["resurrection"]
    model = load_checkpoint(build_model(spec.model_config), str(registry.ensure_weights(spec)))
    x = torch.from_numpy(synthetic_mix(spec.chunk_size / 44100 + 0.1)[:, : spec.chunk_size]).unsqueeze(0)
    fast = None
    if kind == "ref":

        def fn():
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(opt.get("autocast"))):
                return model(x).float()
    else:
        fast = FastBSRoformer(
            model,
            dtype=torch.bfloat16 if opt["dtype"] == "bf16" else torch.float32,
            glue=opt["glue"],
            attn=opt["attn"],
            fused=opt["fused"],
        )

        def fn():
            return fast(x)

    def call():
        if opt.get("worker"):
            box = {}
            th = threading.Thread(target=lambda: box.setdefault("y", fn()))
            th.start()
            th.join()
            return box["y"]
        return fn()

    y = call()  # warm-up (allocations, kernel tables, oneDNN primitive cache)
    times = []
    for _ in range(repeats):
        if fast is not None:
            fast.timer.enabled = True
            fast.timer.t.clear()
        t0 = time.perf_counter()
        y = call()
        times.append(time.perf_counter() - t0)
    np.save(out_dir / f"{name}.npy", y[0].numpy())
    res = {
        "stage": name,
        "description": desc,
        "threads": threads,
        "sec_per_chunk": round(min(times), 3),
        "all": [round(t, 3) for t in times],
    }
    if fast is not None:
        res["profile"] = {k: round(v, 3) for k, v in sorted(fast.timer.t.items(), key=lambda kv: -kv[1])}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--stage", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", help=argparse.SUPPRESS)
    ap.add_argument("--json", help="write all results to this JSON file")
    a = ap.parse_args()

    if a.stage:  # child process
        print("RESULT " + json.dumps(run_stage(a.stage, a.threads, a.repeats, Path(a.out_dir))), flush=True)
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from _common import snr_db

    results = []
    with tempfile.TemporaryDirectory() as td:
        for st in a.stages:
            cmd = [
                sys.executable,
                __file__,
                "--stage",
                st,
                "--threads",
                str(a.threads),
                "--repeats",
                str(a.repeats),
                "--out-dir",
                td,
            ]
            p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            if p.returncode or not line:
                print(f"stage {st} failed:\n{p.stderr[-2000:]}", file=sys.stderr)
                continue
            r = json.loads(line[-1][7:])
            results.append(r)
            print(f"{st:>18}: {r['sec_per_chunk']:7.2f} s/chunk", flush=True)
        ref_path = Path(td) / "stock.npy"
        if ref_path.exists():
            ref = np.load(ref_path)
            for r in results:
                r["snr_vs_stock_fp32_db"] = round(snr_db(ref, np.load(Path(td) / f"{r['stage']}.npy")), 1)
    chunk_audio_s = 749259 / 44100
    print("\n| stage | s / chunk | chunk RTF | SNR vs stock fp32 |\n|---|---:|---:|---:|")
    for r in results:
        snr = r.get("snr_vs_stock_fp32_db", "")
        print(f"| {r['description']} | {r['sec_per_chunk']:.2f} | {r['sec_per_chunk'] / chunk_audio_s:.3f} | {snr} |")
    for r in results:
        if "profile" in r:
            print(f"\nprofile {r['stage']}: " + ", ".join(f"{k} {v:.2f}" for k, v in r["profile"].items()))
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
