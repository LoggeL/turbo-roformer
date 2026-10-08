# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-08

First public release.

### Added
- `turbo_roformer.Separator`: load once, `separate(mix, sr, overlap, progress)` for arrays of any channel count and
  sample rate, `separate_file(...)` for files; vocals = mix - instrumental.
- `turbo-roformer` CLI (WAV/FLAC/MP3/OGG output, `--info` for CPU/backend diagnostics, `--json` results).
- Restructured BS-RoFormer forward (folded norms and attention scale, fused qkv+gates GEMM, precomputed rotary
  tables, no axial rearrange copies, batched mask estimator); exact in fp32 (>= 100 dB vs the reference forward).
- AVX512-BF16 C++ kernels (custom exact-softmax attention, fused GEMM epilogues), compiled on first use with
  `-march=native`; automatic fallback to the portable torch path with a log line.
- Model registry with checksum-verified download of "BS Roformer Instrumental Resurrection" by unwa from the
  author's upload (fallback: python-audio-separator's mirror). No weights are redistributed.
- glibc allocator tuning that removes per-layer mmap/page-fault churn (Linux).
- Tests: reference parity, kernel parity (main and worker thread), overlap-add reconstruction and click checks,
  edge cases (mono, 5.1, 1 sample, odd lengths, 22.05/32/48 kHz, NaN/inf), registry/download, CLI.
- Benchmarks: synthetic 4:12 song benchmark, optimisation ladder, file comparison, MP3 codec floor.
