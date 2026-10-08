"""CLI smoke tests with a tiny random checkpoint (no download)."""

from __future__ import annotations

import json
import subprocess
import sys

import soundfile as sf
import torch
import yaml

from .conftest import TINY_CFG, TINY_DIM_T, make_tiny_model, synth_music


def _tiny_ckpt(tmp_path):
    ckpt = tmp_path / "tiny.ckpt"
    torch.save(make_tiny_model().state_dict(), ckpt)
    cfg = {"model": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TINY_CFG.items()},
           "inference": {"dim_t": TINY_DIM_T}}
    conf = tmp_path / "tiny.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    return ckpt, conf


def _run(*args, check=True):
    return subprocess.run([sys.executable, "-m", "turbo_roformer", *args], capture_output=True, text=True,
                          check=check, timeout=600)


def test_cli_info():
    r = _run("--info")
    assert "turbo-roformer" in r.stdout and "avx512-bf16 kernels" in r.stdout


def test_cli_separates_file(tmp_path):
    ckpt, conf = _tiny_ckpt(tmp_path)
    wav = tmp_path / "song.wav"
    sf.write(wav, synth_music(44100).T, 44100, subtype="PCM_16")
    out = tmp_path / "out"
    r = _run(str(wav), "-o", str(out), "--model", str(ckpt), "--config", str(conf), "--backend", "torch",
             "--precision", "fp32", "--threads", "2", "--format", "wav", "--format", "flac", "--json", "-q")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["rtf"] > 0 and res["duration_s"] == 1.0
    for name in ("song_instrumental.wav", "song_vocals.wav", "song_instrumental.flac", "song_vocals.flac"):
        assert (out / name).exists()
    inst, sr = sf.read(out / "song_instrumental.wav")
    assert sr == 44100 and inst.shape == (44100, 2)


def test_cli_human_output(tmp_path):
    ckpt, conf = _tiny_ckpt(tmp_path)
    wav = tmp_path / "a.wav"
    sf.write(wav, synth_music(20000, channels=1).T, 44100)
    r = _run(str(wav), "-o", str(tmp_path / "o"), "--model", str(ckpt), "--config", str(conf), "--backend",
             "torch", "--threads", "2")
    assert "RTF" in r.stdout and "duration" in r.stdout and "wall" in r.stdout
    assert "chunk 1/" in r.stderr


def test_cli_errors(tmp_path):
    assert _run(check=False).returncode == 2
    assert _run("x.wav", "--overlap", "0.5", check=False).returncode == 2
