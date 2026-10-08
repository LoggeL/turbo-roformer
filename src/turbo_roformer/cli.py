"""Command line interface: ``turbo-roformer song.flac -o out/``."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time

from . import __version__


def _info() -> int:
    import torch

    from . import _kernels, registry

    ok, reason = _kernels.support()
    print(f"turbo-roformer {__version__}")
    print(f"python {platform.python_version()}, torch {torch.__version__}, {platform.system()} {platform.machine()}")
    print(f"cpu: {_kernels.cpu_model_name()} ({torch.get_num_threads()} torch threads by default)")
    print(f"avx512-bf16 kernels: {'yes' if ok else 'no'} ({reason})")
    print(f"cache: {_kernels.cache_root()}")
    for spec in registry.MODELS.values():
        print(f"model {spec.name}: {spec.description} [{spec.filename}, by {spec.author}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="turbo-roformer",
        description="Separate vocals and instrumental with BS-RoFormer on the CPU.",
    )
    ap.add_argument("inputs", nargs="*", metavar="INPUT", help="audio files (wav, flac, mp3, ogg; others via ffmpeg)")
    ap.add_argument("-o", "--out-dir", default="separated", help="output directory (default: ./separated)")
    ap.add_argument("--overlap", type=float, default=2.0, help="predictions per sample, >= 1 (default 2)")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--threads", type=int, default=8, help="torch threads; use your physical core count (default 8)")
    ap.add_argument("--backend", choices=["auto", "kernels", "torch"], default="auto")
    ap.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=["wav", "flac", "mp3", "ogg"],
        help="output format, repeatable (default wav = float32)",
    )
    ap.add_argument("--model", default="resurrection", help="registry name or path to a .ckpt (needs --config)")
    ap.add_argument("--config", default=None, help="YAML config for a custom checkpoint")
    ap.add_argument("--model-dir", default=None, help="weights cache (default ~/.cache/turbo-roformer/models)")
    ap.add_argument("--json", action="store_true", help="print one JSON result line per input")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    ap.add_argument("--info", action="store_true", help="print CPU/backend support and exit")
    ap.add_argument("--version", action="version", version=f"turbo-roformer {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.info:
        return _info()
    if not a.inputs:
        ap.error("no input files (or use --info)")
    if a.overlap < 1:
        ap.error("--overlap must be >= 1")
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s", stream=sys.stderr)

    from .separator import Separator

    sep = Separator(
        a.model,
        precision=a.precision,
        threads=a.threads,
        model_dir=a.model_dir,
        backend=a.backend,
        config=a.config,
        progress=not a.quiet,
    )
    print(f"model load {sep.load_s:.1f} s ({sep.backend}, {sep.precision}, {a.threads} threads)", file=sys.stderr)
    formats = tuple(a.formats or ["wav"])
    for path in a.inputs:
        t_chunk = [time.perf_counter()]

        def progress(k: int, n: int, t_chunk=t_chunk) -> None:
            if not a.quiet:
                now = time.perf_counter()
                print(f"  chunk {k}/{n}: {now - t_chunk[0]:.2f} s", file=sys.stderr, flush=True)
                t_chunk[0] = now

        res = sep.separate_file(path, a.out_dir, formats=formats, overlap=a.overlap, progress=progress)
        if a.json:
            print(json.dumps({"input": path, **res}))
        else:
            print(
                f"{path}: duration {res['duration_s']:.1f} s | {res['n_chunks']} chunks | wall {res['wall_s']:.1f} s "
                f"| RTF {res['rtf']:.3f}"
            )
            for p in res["files"].values():
                print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
