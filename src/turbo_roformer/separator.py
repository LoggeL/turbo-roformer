"""High-level API: ``Separator`` loads a model once and separates arrays or files."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import _kernels, audio, registry
from .chunking import chunk_starts, overlap_add, step_for
from .fast import FastBSRoformer
from .model import BSRoformer, build_model, load_checkpoint

log = logging.getLogger("turbo_roformer")

SR = 44100
PEAK = 0.9  # like audio-separator: louder mixes are scaled to this peak before the model (and back after)
BACKENDS = ("auto", "kernels", "torch")
PRECISIONS = ("bf16", "fp32")

_malloc_tuned = False


def tune_malloc() -> bool:
    """glibc: keep large blocks on the heap instead of fresh mmaps (process-wide, Linux only).

    Every activation > 32 MB would otherwise be mmapped, page-faulted and kernel-zeroed on each layer and
    unmapped on free (measured: the stock forward goes from 40.3 to 31.4 s per chunk with this alone). Same
    effect as ``MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=2000000000``. Disable with
    ``TURBO_ROFORMER_NO_MALLOPT=1``.
    """
    global _malloc_tuned
    if _malloc_tuned:
        return True
    if not sys.platform.startswith("linux") or os.environ.get("TURBO_ROFORMER_NO_MALLOPT"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        ok = libc.mallopt(-4, 0) == 1  # M_MMAP_MAX
        ok = (libc.mallopt(-1, 2_000_000_000) == 1) and ok  # M_TRIM_THRESHOLD
        _malloc_tuned = ok
    except (OSError, AttributeError):
        return False
    return _malloc_tuned


class Separator:
    """Vocal/instrumental separation with a BS-RoFormer model on the CPU.

    Args:
        model: registry name (``"resurrection"``) or a path to a BS-RoFormer ``.ckpt`` (then ``config`` is needed).
        precision: ``"bf16"`` (default; fast kernels) or ``"fp32"`` (exact restructured forward).
        threads: torch intra-op threads (process-wide setting). Use the number of physical cores.
        model_dir: where weights are cached (default ``~/.cache/turbo-roformer/models``).
        backend: ``"auto"`` (kernels if the CPU has AVX512-BF16 and precision is bf16, else torch),
            ``"kernels"`` (fail if unavailable) or ``"torch"`` (portable restructured torch path).
        config: YAML config (ZFTurbo/audio-separator format) for a custom checkpoint path.

    ``separate`` is serialised by an internal lock: an instance owns preallocated work buffers, so concurrent
    calls on the same instance wait for each other. Calls may come from any thread; on threads other than the
    main thread the bf16 backend uses an allocation-free FF1 kernel (glibc ignores the malloc tuning for
    non-main arenas), which gives the same result up to bf16 rounding.
    """

    def __init__(
        self,
        model: str = "resurrection",
        precision: str = "bf16",
        threads: int | None = 8,
        model_dir: str | os.PathLike | None = None,
        *,
        backend: str = "auto",
        config: str | os.PathLike | None = None,
        progress: bool = True,
    ):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if backend == "kernels" and precision != "bf16":
            raise ValueError("the kernels backend is bf16 only; use precision='bf16' or backend='torch'")
        t0 = time.perf_counter()
        self.threads = threads
        if threads:
            torch.set_num_threads(int(threads))
        self.model_name, net, hop, dim_t = self._load_reference(model, model_dir, config, progress)
        self.hop = hop
        self.chunk = hop * (dim_t - 1)
        self._configure(net, backend, precision)
        self.load_s = time.perf_counter() - t0
        self._lock = threading.Lock()
        log.info(
            "turbo-roformer: %s, backend=%s, precision=%s, threads=%s (load %.1f s)",
            self.model_name,
            self.backend,
            self.precision,
            threads or torch.get_num_threads(),
            self.load_s,
        )

    # ------------------------------------------------------------------ construction helpers
    @staticmethod
    def _load_reference(model, model_dir, config, progress) -> tuple[str, BSRoformer, int, int]:
        p = Path(str(model)).expanduser()
        if p.suffix in (".ckpt", ".pt", ".pth") or p.exists():
            if config is None:
                raise ValueError("a custom checkpoint needs config=<yaml>")
            cfg = registry.load_yaml_config(config)
            mcfg = cfg["model"]
            hop = int(mcfg.get("stft_hop_length", 512))
            dim_t = int(cfg.get("inference", {}).get("dim_t") or cfg.get("audio", {}).get("dim_t") or 801)
            net = load_checkpoint(build_model(mcfg), str(p))
            return p.name, net, hop, dim_t
        spec = registry.resolve_spec(str(model))
        path = registry.ensure_weights(spec, model_dir, progress)
        net = load_checkpoint(build_model(spec.model_config), str(path))
        return spec.name, net, spec.hop_length, spec.dim_t

    def _configure(self, net: BSRoformer, backend: str, precision: str) -> None:
        """Pick the backend and build the fast forward. Sets ``self.backend``, ``self.precision``, ``self.net``."""
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if backend == "kernels" and precision != "bf16":
            raise ValueError("the kernels backend is bf16 only; use precision='bf16' or backend='torch'")
        chosen = "torch"
        if backend in ("auto", "kernels") and precision == "bf16":
            try:
                _kernels.load()
                chosen = "kernels"
            except _kernels.KernelUnavailable as e:
                if backend == "kernels":
                    raise
                log.warning("AVX512-BF16 kernels unavailable (%s) -> using the exact fp32 torch path (slower)", e)
                precision = "fp32"
        tune_malloc()
        if chosen == "kernels":
            self.net = FastBSRoformer(net, dtype=torch.bfloat16, glue="cpp", attn="cpp", fused="best")
        else:
            dtype = torch.bfloat16 if precision == "bf16" else torch.float32
            self.net = FastBSRoformer(net, dtype=dtype, glue="torch", attn="sdpa", fused=())
        self.backend, self.precision = chosen, precision

    @classmethod
    def from_module(
        cls,
        net: BSRoformer,
        hop: int,
        dim_t: int,
        precision: str = "fp32",
        backend: str = "torch",
        threads: int | None = None,
    ) -> Separator:
        """Wrap an already constructed reference model (tests, custom loaders)."""
        self = cls.__new__(cls)
        self.threads = threads
        if threads:
            torch.set_num_threads(int(threads))
        self.model_name, self.hop, self.chunk = "custom", hop, hop * (dim_t - 1)
        self._configure(net.float().eval(), backend, precision)
        self.load_s = 0.0
        self._lock = threading.Lock()
        return self

    # ------------------------------------------------------------------ separation
    def n_chunks(self, n_samples: int, overlap: float = 2.0) -> int:
        """Number of model calls for ``n_samples`` at 44.1 kHz."""
        C = min(self.chunk, n_samples)
        return len(chunk_starts(n_samples, C, step_for(C, overlap)))

    @torch.inference_mode()
    def _separate_44k(self, x: np.ndarray, overlap: float, progress) -> np.ndarray:
        n0 = x.shape[1]
        if n0 < self.chunk and (n0 % self.hop or n0 < 3 * self.hop):
            # Shorter than one chunk: the iSTFT returns hop * (n // hop) samples, so pad with < hop zeros so every
            # sample is predicted; below 3 hops pad to 3 hops (torch.stft's reflect padding needs > n_fft/2).
            target = max(-(-n0 // self.hop) * self.hop, 3 * self.hop)
            x = np.pad(x, ((0, 0), (0, target - n0)))
        peak = float(np.abs(x).max())
        gain = PEAK / peak if peak > PEAK else 1.0
        xt = torch.from_numpy(np.ascontiguousarray(x * np.float32(gain) if gain != 1.0 else x))
        inst = overlap_add(xt, self.chunk, overlap, self.net, progress)
        if gain != 1.0:
            inst.div_(gain)
        return np.ascontiguousarray(inst.numpy()[:, :n0])

    def separate(
        self,
        mix,
        sr: int = SR,
        overlap: float = 2.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Return the instrumental as float32 (2, samples) at ``sr``; vocals = ``to_stereo(mix) - inst``.

        ``mix`` is float (channels, samples) or (samples,) at any sample rate: mono is duplicated, more than two
        channels are downmixed (centre kept), other rates are resampled to 44.1 kHz and back. ``overlap`` is the
        number of predictions per sample (>= 1; 2 is the quality/speed default). ``progress(k, n)`` is called
        after each of the n model chunks. Raises ValueError on NaN/inf, empty or malformed input.
        """
        if not overlap >= 1.0:
            raise ValueError("overlap must be >= 1")
        if int(sr) <= 0:
            raise ValueError("sr must be positive")
        sr = int(sr)
        x = audio.to_stereo(audio.as_channels_first(mix))
        n = x.shape[1]
        if sr != SR:
            x = audio.resample(x, sr, SR, length=max(1, round(n * SR / sr)))
        with self._lock:
            if self.threads:
                torch.set_num_threads(int(self.threads))
            inst = self._separate_44k(x, float(overlap), progress)
        if sr != SR:
            inst = audio.resample(inst, SR, sr, length=n)
        return inst

    def separate_file(
        self,
        in_path: str | os.PathLike,
        out_dir: str | os.PathLike,
        formats: tuple[str, ...] | list[str] = ("wav",),
        overlap: float = 2.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Separate a file and write ``<stem>_instrumental.<fmt>`` and ``<stem>_vocals.<fmt>`` into ``out_dir``.

        Returns ``{"instrumental": path, "vocals": path, "files": {...all paths...}, "duration_s", "read_s",
        "separate_s", "write_s", "wall_s", "rtf", "sample_rate", "n_chunks", "backend", "precision"}`` where
        ``rtf = wall_s / duration_s`` (model load excluded; see ``load_s`` on the instance).
        """
        for fmt in formats:
            if fmt not in audio.output_formats():
                raise ValueError(f"unsupported output format {fmt!r}")
        t0 = time.perf_counter()
        mix, sr = audio.load(in_path)
        t1 = time.perf_counter()
        stereo = audio.to_stereo(audio.as_channels_first(mix))
        inst = self.separate(stereo, sr=sr, overlap=overlap, progress=progress)
        vocals = stereo - inst
        t2 = time.perf_counter()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(in_path).stem
        files: dict[str, str] = {}
        for fmt in formats:
            for name, data in (("instrumental", inst), ("vocals", vocals)):
                p = out / f"{stem}_{name}.{fmt}"
                audio.write(p, data, sr, fmt)
                files[f"{name}.{fmt}"] = str(p)
        t3 = time.perf_counter()
        duration = stereo.shape[1] / sr
        return dict(
            instrumental=files[f"instrumental.{formats[0]}"],
            vocals=files[f"vocals.{formats[0]}"],
            files=files,
            duration_s=duration,
            sample_rate=sr,
            n_chunks=self.n_chunks(round(stereo.shape[1] * SR / sr), overlap),
            read_s=t1 - t0,
            separate_s=t2 - t1,
            write_s=t3 - t2,
            wall_s=t3 - t0,
            rtf=(t3 - t0) / duration,
            backend=self.backend,
            precision=self.precision,
        )
