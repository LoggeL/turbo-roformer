"""Model registry and checksum-verified weight download.

Weights are never redistributed by this project. They are downloaded on first use from the model author's
original upload (with python-audio-separator's release mirror as a fallback) and verified against a pinned
sha256 before they are loaded.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._kernels import cache_root
from .model import DEFAULT_FREQS_PER_BANDS

log = logging.getLogger("turbo_roformer")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    filename: str
    sha256: str
    size: int
    urls: tuple[str, ...]
    model_config: dict[str, Any]
    hop_length: int
    dim_t: int
    author: str
    homepage: str
    aliases: tuple[str, ...] = field(default=())

    @property
    def chunk_size(self) -> int:
        return self.hop_length * (self.dim_t - 1)


# Architecture hyper-parameters of "BS Roformer | Instrumental Resurrection by unwa", taken from the ``model:``
# section of config_bs_roformer_instrumental_resurrection_unwa.yaml as shipped by python-audio-separator (MIT)
# and the author's upload; the band layout is lucidrains' DEFAULT_FREQS_PER_BANDS. Training/loss settings
# (multi-resolution STFT loss, dropout schedules, ...) do not affect inference and are omitted.
_RESURRECTION_CFG: dict[str, Any] = dict(
    dim=256,
    depth=12,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    linear_transformer_depth=0,
    freqs_per_bands=DEFAULT_FREQS_PER_BANDS,
    dim_head=64,
    heads=8,
    attn_dropout=0.0,
    ff_dropout=0.0,
    dim_freqs_in=1025,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    mlp_expansion_factor=4,
)

MODELS: dict[str, ModelSpec] = {
    "resurrection": ModelSpec(
        name="resurrection",
        description="BS-RoFormer Instrumental Resurrection (unwa) - instrumental/vocal split, 44.1 kHz stereo",
        filename="bs_roformer_instrumental_resurrection_unwa.ckpt",
        sha256="16311025a5133ae6411760ccfe9e3e66b31a01d9d8bec0a03fa7ec4bedac7a15",
        size=204483033,
        urls=(
            "https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt",
            "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/"
            "bs_roformer_instrumental_resurrection_unwa.ckpt",
        ),
        model_config=_RESURRECTION_CFG,
        hop_length=441,
        dim_t=1700,
        author="unwa (pcunwa)",
        homepage="https://huggingface.co/pcunwa/BS-Roformer-Resurrection",
        aliases=(
            "bs_roformer_instrumental_resurrection_unwa",
            "bs_roformer_instrumental_resurrection_unwa.ckpt",
            "resurrection-inst",
        ),
    ),
}


def resolve_spec(name: str) -> ModelSpec:
    key = name.lower()
    for spec in MODELS.values():
        if key == spec.name or key in spec.aliases:
            return spec
    raise KeyError(f"unknown model {name!r}; available: {', '.join(MODELS)} (or pass a .ckpt path and config=)")


def default_model_dir() -> Path:
    return cache_root() / "models"


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _verified(path: Path, sha256: str) -> bool:
    """Checksum check with a small marker file so a 200 MB checkpoint is hashed once, not on every load."""
    marker = path.with_name(path.name + ".sha256")
    st = path.stat()
    stamp = f"{sha256} {st.st_size} {st.st_mtime_ns}"
    try:
        if marker.read_text().strip() == stamp:
            return True
    except OSError:
        pass
    if sha256_file(path) != sha256:
        return False
    try:
        marker.write_text(stamp + "\n")
    except OSError:
        pass
    return True


def download(url: str, dest: Path, sha256: str | None = None, size: int | None = None, progress: bool = True) -> Path:
    """Stream ``url`` to ``dest`` (atomic rename), verifying sha256 on the fly. Raises on mismatch."""
    if not url.startswith(("https://", "http://", "file:")):
        raise ValueError(f"refusing to download from {url!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".part{os.getpid()}")
    req = urllib.request.Request(url, headers={"User-Agent": "turbo-roformer"})  # noqa: S310 - scheme checked
    h = hashlib.sha256()
    bar = None
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 - scheme checked
            total = int(r.headers.get("Content-Length") or 0) or size
            if progress and sys.stderr.isatty():
                try:
                    from tqdm import tqdm

                    bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False)
                except ImportError:
                    bar = None
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if sha256 is not None and h.hexdigest() != sha256:
            raise OSError(f"sha256 mismatch for {url}: got {h.hexdigest()}, expected {sha256}")
        os.replace(tmp, dest)
    finally:
        if bar is not None:
            bar.close()
        if tmp.exists():
            tmp.unlink()
    if sha256 is not None:
        st = dest.stat()
        dest.with_name(dest.name + ".sha256").write_text(f"{sha256} {st.st_size} {st.st_mtime_ns}\n")
    return dest


def ensure_weights(spec: ModelSpec, model_dir: str | os.PathLike | None = None, progress: bool = True) -> Path:
    """Return the verified local checkpoint path, downloading it on first use."""
    d = Path(model_dir).expanduser() if model_dir else default_model_dir()
    path = d / spec.filename
    if path.exists():
        if _verified(path, spec.sha256):
            return path
        log.warning("%s exists but its sha256 does not match; downloading again", path)
    errors = []
    for url in spec.urls:
        try:
            log.info("downloading %s (%.0f MB) from %s", spec.filename, spec.size / 1e6, url)
            return download(url, path, spec.sha256, spec.size, progress)
        except Exception as e:  # network error, 404, checksum mismatch -> try the next source
            errors.append(f"{url}: {e}")
            log.warning("download failed: %s", errors[-1])
    raise RuntimeError("could not download model weights:\n  " + "\n  ".join(errors))


def load_yaml_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a ZFTurbo/audio-separator style YAML config safely (``!!python/tuple`` is read as a list)."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, node: tuple(ld.construct_sequence(node)))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_Loader)  # noqa: S506 - SafeLoader subclass
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise ValueError(f"{path}: expected a config with a 'model:' section")
    return cfg
