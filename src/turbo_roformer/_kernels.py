"""CPU feature detection and on-demand compilation of the AVX512-BF16 kernels.

The kernels (``csrc/kernels.cpp``) are compiled on first use with ``torch.utils.cpp_extension.load``
(``-O3 -march=native -fopenmp``) into a per-machine cache directory and registered as
``torch.ops.turbo_roformer.*``. Nothing is compiled at install time, so the wheel stays pure Python.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import sys
import threading
from pathlib import Path

log = logging.getLogger("turbo_roformer")

REQUIRED_FLAGS = ("avx512f", "avx512bw", "avx512vl", "avx512_bf16")
_SRC = Path(__file__).with_name("csrc") / "kernels.cpp"
_lock = threading.Lock()
_loaded = False


class KernelUnavailable(RuntimeError):
    """The AVX512-BF16 kernels cannot be used on this machine (reason in the message)."""


def cache_root() -> Path:
    """``$TURBO_ROFORMER_HOME`` or ``$XDG_CACHE_HOME/turbo-roformer`` (default ``~/.cache/turbo-roformer``)."""
    env = os.environ.get("TURBO_ROFORMER_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "turbo-roformer"


def cpu_flags() -> set[str]:
    """CPU feature flags from /proc/cpuinfo (Linux only; empty set elsewhere)."""
    if not sys.platform.startswith("linux"):
        return set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def cpu_model_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def support() -> tuple[bool, str]:
    """(usable, reason). Checks OS/arch, the AVX-512 feature flags and a C++ compiler; does not compile."""
    if os.environ.get("TURBO_ROFORMER_DISABLE_KERNELS"):
        return False, "disabled by TURBO_ROFORMER_DISABLE_KERNELS"
    if not sys.platform.startswith("linux"):
        return False, f"kernels are built for Linux only (this is {sys.platform})"
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False, f"kernels need x86-64 (this is {platform.machine()})"
    flags = cpu_flags()
    missing = [f for f in REQUIRED_FLAGS if f not in flags]
    if missing:
        return False, f"CPU lacks {', '.join(missing)}"
    cxx = os.environ.get("CXX", "c++")
    if shutil.which(cxx) is None and shutil.which("g++") is None:
        return False, "no C++ compiler found (install g++ or set CXX)"
    return True, "AVX512-BF16 available"


def _build_name() -> str:
    import torch

    h = hashlib.sha256()
    h.update(_SRC.read_bytes())
    h.update(torch.__version__.encode())
    h.update(cpu_model_name().encode())
    h.update(" ".join(sorted(cpu_flags())).encode())
    return f"turbo_roformer_kernels_{h.hexdigest()[:12]}"


def _ninja_on_path() -> None:
    """cpp_extension runs ``ninja`` from PATH; a venv used without activation has it only in its bin dir."""
    import sysconfig

    dirs = [sysconfig.get_path("scripts"), str(Path(sys.executable).parent)]
    try:
        import ninja

        dirs.insert(0, getattr(ninja, "BIN_DIR", ""))
    except ImportError:
        pass
    path = os.environ.get("PATH", "").split(os.pathsep)
    extra = [d for d in dirs if d and d not in path and Path(d, "ninja").exists()]
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + path)


def load(verbose: bool = False) -> None:
    """Compile (once per machine/torch version, cached) and load the kernels. Raises KernelUnavailable."""
    global _loaded
    with _lock:
        if _loaded:
            return
        ok, reason = support()
        if not ok:
            raise KernelUnavailable(reason)
        import torch
        from torch.utils import cpp_extension

        _ninja_on_path()
        name = _build_name()
        build_dir = cache_root() / "kernels" / name
        build_dir.mkdir(parents=True, exist_ok=True)
        if not any(build_dir.glob("*.so")):
            log.info("compiling AVX512-BF16 kernels into %s (one-time, ~10-60 s)", build_dir)
        try:
            cpp_extension.load(
                name=name,
                sources=[str(_SRC)],
                extra_cflags=["-O3", "-march=native", "-fopenmp", "-std=c++20"],
                extra_ldflags=["-fopenmp"],
                build_directory=str(build_dir),
                is_python_module=False,
                verbose=verbose,
            )
        except Exception as e:  # compiler errors, missing ninja, ...
            raise KernelUnavailable(f"kernel build failed: {type(e).__name__}: {str(e).strip()[-800:]}") from e
        if not hasattr(torch.ops.turbo_roformer, "attention"):
            raise KernelUnavailable("kernel library loaded but ops are not registered")
        _loaded = True


def is_loaded() -> bool:
    return _loaded
