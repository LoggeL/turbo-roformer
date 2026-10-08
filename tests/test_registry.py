"""Model registry, checksum-verified downloads and config parsing (no network)."""

from __future__ import annotations

import hashlib
import re

import pytest

from turbo_roformer import registry


def test_registry_entries_are_well_formed():
    for spec in registry.MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert spec.urls and all(u.startswith("https://") for u in spec.urls)
        assert spec.chunk_size == spec.hop_length * (spec.dim_t - 1)
    assert registry.MODELS["resurrection"].chunk_size == 749259
    assert sum(registry.MODELS["resurrection"].model_config["freqs_per_bands"]) == 1025


def test_resolve_aliases():
    assert registry.resolve_spec("resurrection").name == "resurrection"
    assert registry.resolve_spec("bs_roformer_instrumental_resurrection_unwa.ckpt").name == "resurrection"
    with pytest.raises(KeyError):
        registry.resolve_spec("nope")


def test_download_verifies_checksum(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"weights" * 1000)
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "cache" / "w.ckpt"
    with pytest.raises(OSError, match="sha256 mismatch"):
        registry.download(src.as_uri(), dest, "0" * 64, progress=False)
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part*"))
    registry.download(src.as_uri(), dest, good, progress=False)
    assert dest.read_bytes() == src.read_bytes()


def test_ensure_weights_uses_fallback_and_cache(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc" * 4096)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    spec = registry.ModelSpec(
        name="t",
        description="",
        filename="t.ckpt",
        sha256=sha,
        size=src.stat().st_size,
        urls=((tmp_path / "missing.bin").as_uri(), src.as_uri()),
        model_config={},
        hop_length=1,
        dim_t=2,
        author="",
        homepage="",
    )
    p = registry.ensure_weights(spec, tmp_path / "models", progress=False)
    assert p.read_bytes() == src.read_bytes()
    src.unlink()  # second call must not need the network
    assert registry.ensure_weights(spec, tmp_path / "models", progress=False) == p
    p.write_bytes(b"corrupt")  # corrupted cache with no reachable source -> error
    with pytest.raises(RuntimeError, match="could not download"):
        registry.ensure_weights(spec, tmp_path / "models", progress=False)


def test_refuses_odd_schemes(tmp_path):
    with pytest.raises(ValueError):
        registry.download("ftp://example.com/x", tmp_path / "x", None, progress=False)


def test_yaml_config_with_python_tuples(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  dim: 8\n  freqs_per_bands: !!python/tuple\n  - 2\n  - 3\ninference:\n  dim_t: 10\n")
    cfg = registry.load_yaml_config(p)
    assert cfg["model"]["freqs_per_bands"] == (2, 3)
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: !!python/object/apply:os.system ['true']\n")
    with pytest.raises(Exception):  # noqa: B017 - any yaml constructor error
        registry.load_yaml_config(bad)


def test_registry_name_wins_over_same_named_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "resurrection").mkdir()
    assert registry.is_known("resurrection") and registry.is_known("RESURRECTION")
    assert not registry.is_known("tiny.ckpt")
