# Contributing

Thanks for helping make CPU source separation faster. Bug reports with numbers are the most useful
contribution: please include the output of `turbo-roformer --info`, your thread count and the RTF you see.

## Development setup

```bash
git clone https://github.com/LoggeL/turbo-roformer && cd turbo-roformer
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only wheels are much smaller
pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest -q                 # fast suite: random small model, no download (~30 s)
python -m pytest -q --weights       # + real-weight parity tests (downloads ~200 MB once)
pip install -e ".[upstream]"         # optional: also compare against python-audio-separator's module
```

- Kernel tests run only on CPUs with `avx512_bf16` (e.g. AMD Zen 4/5, Intel Sapphire Rapids and newer) and skip
  elsewhere. If you change `csrc/kernels.cpp`, run them on such a machine.
- Numerical changes need a parity argument: the fp32 path must stay >= 100 dB from the reference forward, the
  bf16 kernels >= 40 dB from fp32 (in practice about 52-54 dB).
- Performance changes need before/after numbers from `benchmarks/ladder.py` or `benchmarks/bench_synthetic.py`
  with the hardware stated. Use a quiet machine and report the minimum of several runs.

## Style

`ruff check . && ruff format .` (configured in `pyproject.toml`). Keep the public API (`Separator`, CLI flags)
backwards compatible within a minor version.

## Ground rules for test data

Never commit audio files, model weights or song lyrics. Tests and benchmarks use synthetic signals; real-song
results are reported as numbers only ("a 4:12 commercial song, not included").

## License

By contributing you agree that your contributions are licensed under the MIT License of this repository.
