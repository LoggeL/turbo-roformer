# How it works: from 40 s to 6.3 s per chunk

This is the optimisation story behind turbo-roformer, step by step, with the measurements that motivated each
step. Everything was measured on one machine:

> **AMD Ryzen 7 8845HS** (8 cores / 16 threads, Zen 4, AVX-512 + AVX512_BF16, no AMX), 64 GB DDR5,
> Fedora Linux 43, PyTorch 2.14.1 (CPU wheel), 8 torch threads, GCC 15, `-march=native`.

All "per chunk" numbers are the time for **one model call on a 749 259-sample chunk (16.99 s of audio)**, the
minimum of two runs after a warm-up, from [`benchmarks/ladder.py`](../benchmarks/ladder.py).

## The model in one paragraph

BS-RoFormer ([Lu et al., ICASSP 2024](https://arxiv.org/abs/2309.02612)) works on a complex STFT
(n_fft 2048, hop 441 → 1700 frames per chunk). The 1025 bins × 2 channels × (re, im) are split into **62 bands**,
each band is projected to a 256-dim token, giving a 1700 × 62 grid of tokens. Twelve blocks then alternate
**time attention** (62 sequences of 1700 frames) and **frequency attention** (1700 sequences of 62 bands), each
followed by a 4× feed-forward. 62 per-band MLPs predict a complex mask, which is applied to the STFT and inverted.
The "Instrumental Resurrection" checkpoint by unwa has dim 256, 8 heads × 64, depth 12, 51 M parameters.

## Where the FLOPs are

Per chunk (N = 1700 × 62 = 105 400 tokens, 12 blocks):

| part | GFLOP | share |
|---|---:|---:|
| time attention, QKᵀ + PV (62 × 8 sequences of length 1700) | 4 400 | 44 % |
| feed-forward 256 → 1024 → 256 (time + freq) | 2 650 | 27 % |
| qkv + gate projection 256 → 1544 | 2 000 | 20 % |
| output projection 512 → 256 | 660 | 7 % |
| frequency attention (1700 × 8 sequences of length 62) | 160 | 2 % |
| mask estimator, band split, STFT | ~90 | 1 % |
| **total** | **~10 000** | |

So a chunk is about **10 TFLOP**. At 6.36 s that is ~1.6 TFLOP/s sustained on 8 cores. The time attention alone
is 44 % of the work and, as it turns out, the part where the stock CPU path loses the most.

## The ladder

Reproducible with `python benchmarks/ladder.py` (each stage in a fresh process, same synthetic chunk; SNR against
the unmodified fp32 reference forward):

| # | stage | s / chunk | chunk RTF | SNR vs stock fp32 |
|---:|---|---:|---:|---:|
| 0 | reference forward, fp32 | 36.01 | 2.12 | — |
| 1 | + glibc malloc tuning | 25.22 | 1.48 | identical |
| 1b | + `torch.autocast(bf16)` (for comparison only) | 20.33 | 1.20 | 50.4 dB |
| 2 | restructured forward, fp32 | 24.61 | 1.45 | 132.6 dB |
| 3 | restructured forward, bf16 GEMMs + bf16 SDPA | 18.50 | 1.09 | 56.3 dB |
| 4 | + C++ glue (norm, rotary, packing), torch bf16 SDPA | 12.76 | 0.75 | 56.9 dB |
| 5 | + custom AVX512-BF16 attention kernel | 7.59 | 0.45 | 56.9 dB |
| 6 | + qkv GEMM with fused rotary / packing / gate epilogue | 6.68 | 0.39 | 57.4 dB |
| 7 | + out-projection fused with residual add + next RMSNorm | 6.51 | 0.38 | 57.4 dB |
| 8 | + FF2 fused with residual add + next RMSNorm (**default**) | **6.36** | **0.37** | 58.5 dB |
| 8w | default, called from a worker thread | 6.56 | 0.39 | 58.4 dB |

Stage 0 runs this repository's port of the reference module. The original python-audio-separator module
(einops rearranges, rotary helper, same math) measured slower on the same machine, which is where the headline
**40.3 s → 31.4 s (allocator) → … → 6.25 s** comes from:

| python-audio-separator 0.47 module, same chunk | s / chunk |
|---|---:|
| stock, fp32 | 40.27 |
| + malloc tuning (`MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=2000000000`) | 31.35 |
| + `torch.autocast(bf16)` | 21.33 |
| turbo-roformer default, measured inside a full-song run (183 s / 29 chunks incl. overlap-add) | 6.25–6.31 |

### Step 1 – the allocator (36.0 → 25.2 s, free)

glibc serves every allocation above the dynamic mmap threshold (at most 32 MB) with a fresh `mmap` and returns it
with `munmap` on free. A BS-RoFormer layer on a full chunk produces activations of 100–400 MB, so every layer
page-faults and kernel-zeroes about a gigabyte of fresh memory, then gives it back. The stock CLI run on a 4:12 song
recorded **1.12 billion minor page faults**; the final pipeline needs 0.78 million.

`mallopt(M_MMAP_MAX, 0)` and a 2 GB `M_TRIM_THRESHOLD` keep the big blocks on the heap where they are reused.
turbo-roformer does this once at start-up on Linux (opt out with `TURBO_ROFORMER_NO_MALLOPT=1`). Two side notes:
it only applies to the main arena (worker threads get their own arenas with their own rules, see *Threads* below),
and the fast path also keeps its own preallocated work buffers.

### Step 2 – restructure the forward, stay exact (24.6 s, 132.6 dB)

[`fast.py`](../src/turbo_roformer/fast.py) computes the same function with fewer, larger operations:

* **Fold the norms.** Every RMSNorm is `x / ‖x‖ · √d · γ` followed by a Linear; `√d · γ` is folded into the
  Linear's weight columns, so the norm becomes a plain L2 normalisation.
* **Fold the attention scale.** 1/√64 = 1/8 is a power of two, so multiplying it into W_q is exact even in bf16.
* **One GEMM for q, k, v and the gates** (256 → 3·512 + 8 = 1544 outputs).
* **Precomputed rotary tables** in fp32 (exactly the reference angles), applied to interleaved pairs without
  `rotate_half`/`cat`/negation temporaries.
* **No axial rearranges.** Tokens stay in (time, band, dim) order for the whole network; time attention and
  frequency attention just view the same buffer with different strides instead of transposing 100 MB tensors
  twice per block.
* **Batched mask estimator.** Consecutive bands of equal width share one `baddbmm`: 62 small MLPs become 7
  grouped matrix multiplies.

In fp32 this is 132.6 dB from the reference: the differences are float rounding only. Note it is barely faster
than the tuned stock path: the attention (`attn_block` 17.9 s of 24.6 s) still dominates.

### Step 3 – bf16 GEMMs (18.5 s)

Zen 4 executes `vdpbf16ps` (bf16 pairs → fp32 accumulate), so oneDNN's bf16 GEMMs are roughly twice as fast as
fp32. The residual stream stays fp32; only GEMM operands are bf16. The feed-forward drops from 6.5 s to 2.4 s
(oneDNN with the GELU fused as a post-op). But the profile now reads `attn_block` **16.1 s of 18.5 s**.

### Step 4 – C++ glue (12.8 s) and why bf16 SDPA is slow here

Small fused kernels (`add_rmsnorm`, `rope_pack`, `gate_merge`) remove the remaining elementwise passes and copies
around the attention. What is left is plain: **time attention 7.19 s** for 4.4 TFLOP, i.e. ~0.6 TFLOP/s, while
the GEMMs around it run at 1.3–2 TFLOP/s.

PyTorch's CPU `scaled_dot_product_attention` for bf16 is a flash-attention style kernel: it tiles the keys,
keeps an online softmax in fp32 and converts blocks between bf16 and fp32 as it goes. As far as we can tell from
the PyTorch sources, its packed bf16 micro-kernel path (oneDNN brgemm) is taken only on CPUs with **AMX** (Intel
Sapphire Rapids and newer). Zen 4 has AVX512-BF16 (`vdpbf16ps`) but no AMX tiles, so the block products run
through a generic GEMM call per small block, and the per-block conversion and rescaling overhead is large
relative to the 64-wide head dimension. We did not profile PyTorch's kernel further; the observable fact is
~0.6 TFLOP/s for the time attention through bf16 SDPA versus ~1.6 TFLOP/s for a kernel built around the same
instruction (next step), see also the [instruction throughput](benchmarks.md#instruction-throughput).

### Step 5 – a custom attention kernel (7.6 s)

[`csrc/kernels.cpp`](../src/turbo_roformer/csrc/kernels.cpp) `attention()`:

* **Layouts made for `vdpbf16ps`.** K is stored as 64-key tiles of 32 bf16 *pairs* along the head dimension
  (VNNI layout, 8 KB per tile), V as key-pairs interleaved per output column. A query pair is broadcast and
  multiplied against 16 keys per instruction.
* **Register blocking.** 6 query rows × 64 keys = 24 zmm accumulators for QKᵀ; the same 6 × 64 shape for PV.
* **Exact softmax, not online.** A row of scores is at most 1700 floats, so all 6 rows (40 KB) stay in L1/L2: pass
  one computes the scores and the running max (fused into the QKᵀ tile loop), pass two computes `2^(s·log2e − m)`
  with a degree-4 polynomial (relative error 4·10⁻⁵, far below the bf16 step it is rounded to), stores
  unnormalised bf16 probabilities and sums in fp32. No rescaling of partial outputs is ever needed.
* **Fused epilogue.** The 1/sum normalisation, the sigmoid gate of the head, the head merge and the scatter back
  into (time, band) token order happen when the 6 × 64 output tile is written; the result goes straight into the
  bf16 input of the output projection.
* **Dynamic scheduling** over (band, head) sequences with an atomic counter on top of `at::parallel_for`, so a
  core that is briefly interrupted does not stall the whole layer.

Time attention: 7.19 s → **2.68 s** (~1.6 TFLOP/s); frequency attention 0.73 s → 0.15 s.

### Steps 6–8 – fused GEMM epilogues (6.36 s)

With attention fixed, the remaining waste was memory traffic between GEMMs and elementwise kernels. Three custom
GEMMs (6 × 64 register tiles of `vdpbf16ps`, weights pre-packed into VNNI tiles at load time) do the elementwise
work while the accumulators are still in registers:

* `qkv_rope_gemm`: q/k get the rotary rotation in fp32 and are written directly in the attention kernel's packed
  layouts; v is pair-interleaved; the 8 gate columns get bias + sigmoid. (1.53 + 0.50 s → 1.15 s)
* `gemm_res_norm` for the output projection and for FF2: `X += A·Wᵀ (+b)` into the fp32 residual stream, and the
  next layer's L2 norm is computed from the same registers and written as the bf16 input of the next GEMM.
* FF1 stays on oneDNN with its fused GELU post-op (0.88 s); a custom GEMM+GELU kernel exists (1.04 s) and is used
  only off the main thread (see below).

### Threads

`separate()` may be called from any thread. glibc's per-thread arenas serve blocks above 64 MB with fresh mmaps
regardless of `mallopt`, so on a worker thread the 216 MB oneDNN FF1 output would bring back the page-fault churn.
On non-main threads the fused path therefore switches to the allocation-free `gemm_gelu` kernel: 6.56 s instead
of 6.36 s per chunk, same result up to bf16 rounding (tested). One instance processes one call at a time (an
internal lock serialises concurrent calls) because it owns preallocated work buffers.

Thread count: 8 threads (one per physical core) is the sweet spot on the 8845HS; SMT siblings add nothing for
these kernels (16 threads: 6.43 s median per chunk), 6 threads give 8.0 s, 4 threads 11.5 s.

## Numerics

| quantity | precision |
|---|---|
| STFT, band split, mask estimator, complex mask, iSTFT | fp32 |
| residual stream, norms, softmax statistics, rotary math, all accumulators | fp32 |
| GEMM / attention operands (activations, weights, probabilities) | bf16 (round-to-nearest-even) |

Measured differences: 52–58 dB SNR between the bf16 kernels and exact fp32 on single chunks (synthetic input),
53.7 dB on a full real song. For scale: the chunked inference itself (overlap 2 vs. overlap 8) differs from a
"perfect" reference by ~31 dB, so the bf16 error is more than 20 dB below the method's own inherent error, and
the quality versus an overlap-8 fp32 reference is unchanged (31.34 dB, vs 31.37 dB for stock
python-audio-separator and 31.36 dB for exact fp32). An MP3 encode of the result adds far more error than bf16 does (see
[benchmarks](benchmarks.md#mp3-caveat)).

## What did not make it

* **OpenVINO** (exported core network, bf16, best variant): 8.1 s per chunk, song RTF 0.95 — good, but the
  hand-written attention is faster.
* **ONNX Runtime** fp32: 30.7 s per chunk.
* **The integrated GPU** (Radeon 780M via ROCm): unstable on this machine (GPU hangs), not pursued.
* **Mel-band RoFormer** models: a pre-release experiment (not in this repository) reusing the generic kernels
  reached ~0.36 chunk RTF (~61 dB vs fp32) but is not packaged yet — see the roadmap in the README.
