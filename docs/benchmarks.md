# Benchmarks

All numbers on one machine unless stated otherwise:

> **AMD Ryzen 7 8845HS** (Zen 4, 8 cores / 16 threads, AVX-512 + AVX512_BF16, no AMX), 64 GB DDR5,
> Fedora Linux 43 (kernel 7.2), Python 3.12, PyTorch 2.14.1+cpu, GCC 15.3, 8 torch threads, nice 10, otherwise idle
> but not isolated (it is a home server running other services; repeat runs vary by ~1–2 %).

turbo-roformer 0.1.0, model "BS Roformer Instrumental Resurrection" (unwa), overlap 2 unless stated.

## Methodology

- **RTF** (real-time factor) = wall-clock time of `separate`/`separate_file` divided by the audio duration.
  Model loading (~0.4–1 s from the local cache) is excluded; reading and writing the WAV files are included in
  the CLI numbers. RTF < 1 means faster than real time.
- **Per-chunk time** = one model call on 749 259 samples (16.99 s), the unit of work. A 4:12 song at overlap 2
  needs 29 calls; overlap 1.5 needs 22, overlap 3 needs 43.
- **Quality** is measured **model-vs-model**, never against ground-truth stems: we only claim to compute the same
  function faster. Metrics: SNR over the whole song, scale-invariant SNR, and the worst / median SNR over 1-s
  windows whose reference RMS is above −50 dBFS ([`benchmarks/compare_wavs.py`](../benchmarks/compare_wavs.py)).
- **References**: (a) exact fp32 (`--precision fp32`, ≥ 100 dB from the reference forward); (b) stock
  python-audio-separator 0.47.0 CLI output (fp32, same overlap); (c) an overlap-8 fp32 separation of the same song
  computed independently on another machine (Apple M-series, PyTorch MPS, audio-separator) — a stand-in for "the
  model with as little chunking error as is practical".
- **Real song**: a 4:12 commercial pop/worship song, 44.1 kHz stereo, *not included* (copyright). Everything that
  does not need a real song uses the deterministic synthetic signal from
  [`benchmarks/_common.py`](../benchmarks/_common.py); runtime does not depend on content.

## End-to-end: real song (4:12)

`turbo-roformer song.wav -o out --overlap N --json`, wall clock and peak RSS from `/usr/bin/time -v`:

| pipeline | overlap | chunks | wall | RTF | peak RSS |
|---|---:|---:|---:|---:|---:|
| python-audio-separator 0.47.0 CLI, fp32 (stock) | 2 | 29 | 20:32 | 4.88 | 3.0 GB |
| turbo-roformer, exact fp32 path (`--backend torch --precision fp32`) ¹ | 2 | 29 | 12:19 | 2.93 | 3.67 GB |
| **turbo-roformer, bf16 kernels (default)** | **2** | **29** | **3:05** | **0.732** | **2.39 GB** |
| turbo-roformer, bf16 kernels | 1.5 | 22 | 2:20 | 0.556 | 2.39 GB |
| turbo-roformer, bf16 kernels | 3 | 43 | 4:34 | 1.087 | 2.39 GB |

¹ re-measured with v0.1.0 (the pre-release code it was ported from measured 12:17, RTF 2.92); the output is
bit-identical to that pre-release fp32 output.
Earlier runs of the same default configuration measured RTF 0.717–0.733 (181–185 s).

Quality of the same runs:

| comparison | SNR | SI-SNR | worst 1 s | median 1 s |
|---|---:|---:|---:|---:|
| bf16 (default) vs exact fp32 | 53.69 dB | 53.69 dB | 38.5 dB | 54.7 dB |
| bf16 (default) vs stock audio-separator | 53.64 dB | 53.67 dB | 37.8 dB | 54.4 dB |
| exact fp32 vs stock audio-separator | 73.69 dB | 75.99 dB | 45.8 dB | 72.0 dB |
| stock audio-separator vs overlap-8 reference | 31.37 dB | 31.37 dB | 19.6 dB | 32.3 dB |
| exact fp32 vs overlap-8 reference | 31.36 dB | 31.37 dB | 19.5 dB | 32.3 dB |
| **bf16, overlap 2 vs overlap-8 reference** | **31.34 dB** | 31.35 dB | 19.5 dB | 32.2 dB |
| bf16, overlap 1.5 vs overlap-8 reference | 29.61 dB | 29.62 dB | 16.5 dB | 30.2 dB |
| bf16, overlap 3 vs overlap-8 reference | 33.28 dB | 33.28 dB | 20.1 dB | 34.0 dB |

Reading: the bf16 kernels move the result by ~54 dB, the chunking (overlap 2 vs 8) by ~31 dB. Precision is not
what limits quality; the overlap is. Overlap 1.5 trades 1.7 dB for 24 % less time, overlap 3 buys 1.9 dB for
48 % more.

The package's output is **bit-identical** (`max |diff| = 0`) to the pre-release script it was ported from, on the
full song.

## Synthetic song (reproducible)

`python benchmarks/bench_synthetic.py [--seconds S] [--threads T] [--worker-thread] [--quality-seconds Q]`

| run | chunks | median s/chunk | wall | RTF | peak RSS |
|---|---:|---:|---:|---:|---:|
| 252.39 s, main thread | 29 | 6.33 | 183.0 s | 0.725 | 2.29 GB |
| 252.39 s, worker thread | 29 | 6.59 | 191.3 s | 0.758 | 2.59 GB |
| 60 s, 4 threads | 7 | 11.47 | 80.7 s | 1.35 | 2.13 GB |
| 60 s, 6 threads | 7 | 8.04 | 56.3 s | 0.94 | 2.07 GB |
| 60 s, 8 threads | 7 | 6.37 | 44.8 s | 0.75 | 2.14 GB |
| 60 s, 16 threads (SMT) | 7 | 6.43 | 44.9 s | 0.75 | 2.15 GB |

Short songs have a slightly higher RTF because the last chunk is aligned to the end and overlaps more.
bf16 vs exact fp32 on the first 34 s of the synthetic song: **58.8 dB**.

## Per-chunk ladder

`python benchmarks/ladder.py` (fresh process per stage, min of 2 runs after warm-up):

| stage | s / chunk | chunk RTF | SNR vs stock fp32 |
|---|---:|---:|---:|
| reference forward, fp32 | 36.01 | 2.119 | — |
| + glibc malloc tuning | 25.22 | 1.484 | identical |
| + torch.autocast bf16 | 20.33 | 1.196 | 50.4 dB |
| restructured forward, fp32 (exact) | 24.61 | 1.448 | 132.6 dB |
| restructured forward, bf16 GEMMs + bf16 SDPA | 18.50 | 1.089 | 56.3 dB |
| + C++ norm/rotary/packing glue (torch bf16 SDPA) | 12.76 | 0.751 | 56.9 dB |
| + custom AVX512-BF16 attention kernel | 7.59 | 0.447 | 56.9 dB |
| + fused qkv GEMM with rotary/packing epilogue | 6.68 | 0.393 | 57.4 dB |
| + fused out-proj + residual + RMSNorm | 6.51 | 0.383 | 57.4 dB |
| + fused FF2 + residual + RMSNorm (default) | 6.36 | 0.374 | 58.5 dB |
| default, called from a worker thread | 6.56 | 0.386 | 58.4 dB |

The python-audio-separator module itself (not our port) measured 40.27 s stock, 31.35 s with the malloc
environment variables and 21.33 s with autocast bf16 (50.2 dB) on the same chunk.

### Profiles (seconds per chunk, from `FastBSRoformer.timer`)

| section | restructured fp32 | restructured bf16 | C++ glue + SDPA | + attention kernel | default |
|---|---:|---:|---:|---:|---:|
| time attention | (in attn block) | (in attn block) | 7.19 | 2.68 | 2.69 |
| frequency attention | | | 0.73 | 0.15 | 0.17 |
| qkv GEMM (+ rotary/packing) | | | 1.61 + 0.49 | 1.53 + 0.50 | 1.18 (fused) |
| output projection | | | 0.40 | 0.37 | 0.40 (+ residual/norm fused) |
| FF1 (oneDNN GEMM + GELU) | | | 0.88 | 0.86 | 0.88 |
| FF2 | | | 0.81 | 0.80 | 0.75 (+ residual/norm fused) |
| residual add + RMSNorm (unfused kernels) | | | 0.37 | 0.37 | 0.00 |
| attention blocks total | 17.89 | 16.09 | | | |
| feed-forward blocks total | 6.46 | 2.40 | | | |
| mask estimator | 0.22 | 0.23 | 0.22 | 0.29 | 0.24 |
| STFT + band split + iSTFT | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 |

### Instruction throughput

[`benchmarks/dpbf16_peak.cpp`](../benchmarks/dpbf16_peak.cpp), one core, 12 independent accumulators:

| instruction | Ginstr/s | GFLOP/s (1 core) |
|---|---:|---:|
| `vdpbf16ps zmm` (32 bf16 MACs) | 4.90 | 313 |
| `vdpbf16ps ymm` | 8.31 | 266 |
| `vfmadd ps zmm` (16 fp32 MACs) | 4.89 | 157 |

So bf16 dot products give 2× the fp32 FMA rate on Zen 4. Eight cores at the single-core clock would be
~2.5 TFLOP/s; the default pipeline sustains ~1.6 TFLOP/s over the whole chunk (~10 TFLOP) and the attention
kernel ~1.6 TFLOP/s on its 4.4 TFLOP — roughly two thirds of that optimistic peak (all-core clocks are lower than
the single-core clock, so the real fraction is higher).

## Other runtimes tried (same machine, same model)

| runtime | precision | s / chunk | song RTF | SNR vs overlap-8 ref |
|---|---|---:|---:|---:|
| OpenVINO (exported core, NeoX-rotary variant), 1 stream | bf16 | 8.13–8.25 | 0.953 | 31.34 dB |
| OpenVINO, 2 streams (THROUGHPUT hint) | bf16 | ~7.8 effective | 0.938 | 31.34 dB |
| OpenVINO | fp32 | 16.12 | 1.86 | 31.37 dB |
| ONNX Runtime (legacy exporter, opset 17) | fp32 | 30.7 | — | — |
| **turbo-roformer kernels** | bf16 | **6.25–6.36** | **0.72–0.73** | **31.34 dB** |

OpenVINO is the strongest off-the-shelf option on this CPU (and not tied to AVX512-BF16), but needs a model
export that does not trace cleanly from the original module. It is not part of this package.

## MP3 caveat

If the result is delivered as MP3, the codec dominates the error budget. Encoding the bf16 instrumental with
libmp3lame (CBR, `benchmarks/codec_floor.py`) and decoding it again:

| bitrate | SNR(decoded, original) |
|---|---:|
| 128 kbit/s | 20.2 dB |
| 192 kbit/s | 25.6 dB |
| 320 kbit/s | 36.5 dB |

The fp32 stock output gives the same numbers (20.2 / 36.5 dB). Even a 320 kbit/s MP3 changes the signal ~17 dB more
than bf16 does (53.7 dB), so for MP3 deliverables the precision choice is irrelevant. Note that SNR is a waveform
metric: a perceptual codec is designed to be inaudible at much lower SNR, so this table says nothing about
audible quality — only that bf16 rounding is far below the noise floor that the delivery format adds anyway.

## Reproducing

```bash
python benchmarks/ladder.py --json ladder.json                    # ~7 min
python benchmarks/bench_synthetic.py                              # 4:12 synthetic song, ~3 min
python benchmarks/bench_synthetic.py --seconds 60 --quality-seconds 34
turbo-roformer your_song.wav -o out --json                        # your own music
python benchmarks/compare_wavs.py reference.wav out/your_song_instrumental.wav
python benchmarks/codec_floor.py out/your_song_instrumental.wav
```

Please include `turbo-roformer --info` with any numbers you report.
