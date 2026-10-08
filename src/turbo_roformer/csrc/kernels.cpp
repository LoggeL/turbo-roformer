// turbo-roformer: fused CPU kernels for BS-RoFormer inference (x86-64, AVX-512F/BW + AVX512_BF16).
//
// SPDX-License-Identifier: MIT
//
//   add_rmsnorm   : X += Y (optional, bf16/fp32) ; out = X / max(||X||, eps)   (RMSNorm gamma folded into next Linear)
//   rope_pack     : qkv+gate GEMM output -> rotary(q,k) in fp32 math, bf16 Q, K/V either plain or VNNI-packed,
//                   sigmoid(gates); handles the time/freq axial grouping without any rearrange copies
//   attention     : exact (non-online) softmax attention, bf16 dot products (vdpbf16ps) with fp32 accumulation,
//                   gate multiply + head merge + token-order scatter fused into the epilogue
//   gate_merge    : gate multiply + head merge for the torch-SDPA variant
//   qkv_rope_gemm : xn @ [Wq|Wk|Wv|Wg]^T with rotary, VNNI packing and sigmoid gates fused into the GEMM epilogue
//   gemm_res_norm : X += A @ W^T (+b) ; xn = X / ||X||  (residual add + next RMSNorm fused into the GEMM epilogue)
//   gemm_gelu     : H = gelu(xn @ W1^T + b1)  (allocation-free FF1, used off the main thread)
//
// Built at first use by turbo_roformer/_kernels.py via torch.utils.cpp_extension with -O3 -march=native -fopenmp
// (at::parallel_for is header-inlined: without -fopenmp it silently runs single-threaded) and registered as
// torch.ops.turbo_roformer.*.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

using at::Tensor;

static inline float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) {  // round to nearest even (no NaN in this pipeline)
  uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t)(u >> 16);
}
static inline __m512 load_bf16x16(const uint16_t* p) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}
static inline void store_bf16x16(uint16_t* p, __m512 v) {
  __m256bh h = _mm512_cvtneps_pbh(v);
  _mm256_storeu_si256((__m256i*)p, (__m256i)h);
}

// expf for x <= 0 (cephes polynomial, ~1 ulp)
static inline __m512 exp512(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-87.3f));
  __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
  __m512 p = _mm512_set1_ps(1.9875691500E-4f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507E-3f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073E-3f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894E-2f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459E-1f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201E-1f));
  p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r);
  p = _mm512_add_ps(p, _mm512_set1_ps(1.0f));
  return _mm512_scalef_ps(p, n);
}

// exp for softmax probabilities that are rounded to bf16 afterwards: 2^f via degree-4 Taylor on |f|<=0.5
// (rel. error <= 4.2e-5, i.e. ~100x below the bf16 rounding step), x clamped at -80 (no denormals).
static inline __m512 exp512_fast(__m512 x) {
  const __m512 t = _mm512_mul_ps(_mm512_max_ps(x, _mm512_set1_ps(-80.0f)), _mm512_set1_ps(1.44269504088896341f));
  const __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 f = _mm512_sub_ps(t, n);
  __m512 p = _mm512_set1_ps(9.6181291e-3f);
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504109e-2f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022651e-1f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718e-1f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
  return _mm512_scalef_ps(p, n);
}

// 2^t for t <= 0 (same polynomial as exp512_fast, caller supplies t = x*log2e already shifted), clamp at -120
static inline __m512 exp2_512_fast(__m512 t) {
  t = _mm512_max_ps(t, _mm512_set1_ps(-120.0f));
  const __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 f = _mm512_sub_ps(t, n);
  __m512 p = _mm512_set1_ps(9.6181291e-3f);
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504109e-2f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022651e-1f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718e-1f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
  return _mm512_scalef_ps(p, n);
}

// Dynamic scheduling on top of at::parallel_for (shared, noisy machine -> avoid static imbalance)
template <class Fn>
static void dyn_parallel(int64_t n, int64_t chunk, const Fn& fn) {
  std::atomic<int64_t> next{0};
  int nt = at::get_num_threads();
  at::parallel_for(0, nt, 1, [&](int64_t, int64_t) {
    for (;;) {
      int64_t b = next.fetch_add(chunk);
      if (b >= n) break;
      fn(b, std::min(n, b + chunk));
    }
  });
}

// ---------------------------------------------------------------- add_rmsnorm
void add_rmsnorm(Tensor X, const std::optional<Tensor>& Y, Tensor out, double eps) {
  TORCH_CHECK(X.is_contiguous() && X.scalar_type() == at::kFloat && X.dim() == 2);
  TORCH_CHECK(out.is_contiguous() && out.sizes() == X.sizes());
  const int64_t N = X.size(0), D = X.size(1);
  TORCH_CHECK(D % 16 == 0);
  float* x = X.data_ptr<float>();
  const bool hasY = Y.has_value() && Y->defined();
  const bool ybf = hasY && Y->scalar_type() == at::kBFloat16;
  if (hasY) TORCH_CHECK(Y->is_contiguous() && Y->sizes() == X.sizes());
  const void* y = hasY ? Y->data_ptr() : nullptr;
  const bool obf = out.scalar_type() == at::kBFloat16;
  void* o = out.data_ptr();
  const float feps = (float)eps;
  at::parallel_for(0, N, 512, [&](int64_t b, int64_t e) {
    for (int64_t n = b; n < e; ++n) {
      float* xr = x + n * D;
      __m512 acc = _mm512_setzero_ps();
      for (int64_t d = 0; d < D; d += 16) {
        __m512 v = _mm512_loadu_ps(xr + d);
        if (hasY) {
          __m512 w = ybf ? load_bf16x16((const uint16_t*)y + n * D + d) : _mm512_loadu_ps((const float*)y + n * D + d);
          v = _mm512_add_ps(v, w);
          _mm512_storeu_ps(xr + d, v);
        }
        acc = _mm512_fmadd_ps(v, v, acc);
      }
      const float nrm = std::sqrt(_mm512_reduce_add_ps(acc));
      const __m512 vi = _mm512_set1_ps(1.0f / std::max(nrm, feps));
      for (int64_t d = 0; d < D; d += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(xr + d), vi);
        if (obf) store_bf16x16((uint16_t*)o + n * D + d, v);
        else _mm512_storeu_ps((float*)o + n * D + d, v);
      }
    }
  });
}

// ---------------------------------------------------------------- rope_pack
// qkvg : [N, ld] bf16, columns q(h,d) | k(h,d) | v(h,d) | gates(h)   (H=8, D=64)
// cosT/sinT : [L, 32] fp32 rotary tables for interleaved pairs (2p, 2p+1)
// mode 0 = time attention (B=F groups, L=T, token = i*F + b); mode 1 = freq (B=T, L=F, token = b*F + i)
// packed=true : K -> [BH, 32, Lpad] u32 storage, tile-major: [BH][Lpad/64][32 pairs][64 keys] (VNNI pairs of d),
//               V -> [BH, Lpad/2, 64] u32 (key-pair interleaved)
// packed=false: K, V -> [BH, L, 64] bf16 (plain, for torch SDPA)
// One task = one axial group b and 16 consecutive positions, all 8 heads (each token row is read once).

// rotate 16 interleaved pairs (32 bf16) with fp32 math, return 16 packed bf16 pairs
static inline __m512i rope16(const uint16_t* src, const float* c, const float* s) {
  const __m512i v = _mm512_loadu_si512(src);
  const __m512 x0 = _mm512_castsi512_ps(_mm512_slli_epi32(v, 16));
  const __m512 x1 = _mm512_castsi512_ps(_mm512_and_si512(v, _mm512_set1_epi32((int)0xFFFF0000u)));
  const __m512 cc = _mm512_loadu_ps(c), ss = _mm512_loadu_ps(s);
  const __m512 o0 = _mm512_fmsub_ps(x0, cc, _mm512_mul_ps(x1, ss));
  const __m512 o1 = _mm512_fmadd_ps(x1, cc, _mm512_mul_ps(x0, ss));
  const __m512i r0 = _mm512_cvtepu16_epi32((__m256i)_mm512_cvtneps_pbh(o0));
  const __m512i r1 = _mm512_cvtepu16_epi32((__m256i)_mm512_cvtneps_pbh(o1));
  return _mm512_or_si512(r0, _mm512_slli_epi32(r1, 16));
}

std::vector<Tensor> rope_pack(Tensor qkvg, Tensor cosT, Tensor sinT, int64_t mode, int64_t T, int64_t F,
                              int64_t Lpad, bool packed) {
  constexpr int64_t H = 8, D = 64, P = 32;
  TORCH_CHECK(qkvg.scalar_type() == at::kBFloat16 && qkvg.is_contiguous());
  const int64_t N = qkvg.size(0), ld = qkvg.size(1);
  TORCH_CHECK(N == T * F && ld >= 3 * H * D + H);
  const int64_t B = mode == 0 ? F : T, L = mode == 0 ? T : F;
  TORCH_CHECK(cosT.size(0) >= L && cosT.size(1) == P && cosT.is_contiguous() && sinT.is_contiguous());
  if (!packed) Lpad = ((L + 15) / 16) * 16;
  TORCH_CHECK(Lpad % 16 == 0 && Lpad >= L);
  auto bfo = qkvg.options();
  Tensor Q = at::empty({B * H, L, D}, bfo);
  Tensor K, V;
  if (packed) {
    K = at::empty({B * H, P, Lpad}, qkvg.options().dtype(at::kInt));
    V = at::empty({B * H, Lpad / 2, D}, qkvg.options().dtype(at::kInt));
  } else {
    K = at::empty({B * H, L, D}, bfo);
    V = at::empty({B * H, L, D}, bfo);
  }
  Tensor G = at::empty({N, H}, qkvg.options().dtype(at::kFloat));
  const uint16_t* src0 = (const uint16_t*)qkvg.data_ptr();
  const float* cs = cosT.data_ptr<float>();
  const float* sn = sinT.data_ptr<float>();
  uint16_t* q0 = (uint16_t*)Q.data_ptr();
  float* g0 = G.data_ptr<float>();
  uint32_t* k0 = (uint32_t*)K.data_ptr();
  uint32_t* v0 = (uint32_t*)V.data_ptr();
  const int64_t nblk = Lpad / 16;
  dyn_parallel(B * nblk, 4, [&](int64_t tb, int64_t te) {
    alignas(64) uint32_t kt[H][16][P];
    alignas(64) uint16_t vt[H][16][D];
    for (int64_t task = tb; task < te; ++task) {
      const int64_t b = task / nblk, blk = task % nblk;
      for (int64_t jj = 0; jj < 16; ++jj) {
        const int64_t j = blk * 16 + jj;
        if (j >= L) {
          for (int64_t h = 0; h < H; ++h) { std::memset(kt[h][jj], 0, sizeof(kt[h][jj])); std::memset(vt[h][jj], 0, sizeof(vt[h][jj])); }
          continue;
        }
        const int64_t tok = mode == 0 ? j * F + b : b * F + j;
        const uint16_t* src = src0 + tok * ld;
        const float* c = cs + j * P;
        const float* s = sn + j * P;
        for (int64_t h = 0; h < H; ++h) {
          const int64_t bh = b * H + h;
          const uint16_t* qs = src + h * D;
          uint16_t* qd = q0 + (bh * L + j) * D;
          _mm512_storeu_si512(qd, rope16(qs, c, s));
          _mm512_storeu_si512(qd + 32, rope16(qs + 32, c + 16, s + 16));
          const uint16_t* ks = src + H * D + h * D;
          const __m512i ka = rope16(ks, c, s), kb = rope16(ks + 32, c + 16, s + 16);
          const uint16_t* vs = src + 2 * H * D + h * D;
          if (packed) {
            _mm512_storeu_si512(kt[h][jj], ka);
            _mm512_storeu_si512(kt[h][jj] + 16, kb);
            std::memcpy(vt[h][jj], vs, D * 2);
          } else {
            uint16_t* kd = (uint16_t*)k0 + (bh * L + j) * D;
            _mm512_storeu_si512(kd, ka);
            _mm512_storeu_si512(kd + 32, kb);
            std::memcpy((uint16_t*)v0 + (bh * L + j) * D, vs, D * 2);
          }
          g0[tok * H + h] = 1.0f / (1.0f + std::exp(-bf2f(src[3 * H * D + h])));
        }
      }
      if (packed) {
        for (int64_t h = 0; h < H; ++h) {
          const int64_t bh = b * H + h;
          uint32_t* kp = k0 + bh * P * Lpad + (blk / 4) * (P * 64) + (blk % 4) * 16;
          for (int64_t p = 0; p < P; ++p) {
            alignas(64) uint32_t row[16];
            for (int64_t jj = 0; jj < 16; ++jj) row[jj] = kt[h][jj][p];
            _mm512_storeu_si512(kp + p * 64, _mm512_load_si512(row));
          }
          uint32_t* vp = v0 + (bh * (Lpad / 2) + blk * 8) * D;
          for (int64_t jp = 0; jp < 8; ++jp)
            for (int64_t d = 0; d < D; d += 16) {
              const __m512i a = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(vt[h][2 * jp] + d)));
              const __m512i bb = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(vt[h][2 * jp + 1] + d)));
              _mm512_storeu_si512(vp + jp * D + d, _mm512_or_si512(a, _mm512_slli_epi32(bb, 16)));
            }
        }
      }
    }
  });
  return {Q, K, V, G};
}

// ---------------------------------------------------------------- attention
// Q [BH, L, 64] bf16 (scale folded), Kp [BH, 32, Lpad] u32, Vp [BH, Lpad/2, 64] u32, G [N, 8] fp32 (sigmoid gates)
// O [N, 512] bf16 : O[tok, h*64:(h+1)*64] = gate * softmax(q k^T) v
static constexpr int NR = 6;

void attention(Tensor Q, Tensor Kp, Tensor Vp, Tensor G, Tensor O, int64_t mode, int64_t T, int64_t F) {
  constexpr int64_t H = 8, D = 64, P = 32;
  const int64_t BH = Q.size(0), L = Q.size(1), Lpad = Kp.size(2);
  const int64_t B = BH / H;
  TORCH_CHECK(Lpad % 64 == 0 && Lpad >= L && Vp.size(1) == Lpad / 2);
  TORCH_CHECK((mode == 0 && B == F && L == T) || (mode == 1 && B == T && L == F));
  TORCH_CHECK(O.scalar_type() == at::kBFloat16 && O.is_contiguous() && O.size(1) == H * D);
  const uint32_t* q0 = (const uint32_t*)Q.data_ptr();
  const uint32_t* kp0 = (const uint32_t*)Kp.data_ptr();
  const uint32_t* vp0 = (const uint32_t*)Vp.data_ptr();
  const float* g0 = G.data_ptr<float>();
  uint16_t* o0 = (uint16_t*)O.data_ptr();
  const int64_t nrg = (L + NR - 1) / NR;
  const int64_t npair = (L + 1) / 2;
  const int64_t chunk = mode == 0 ? 1 : 16;
  dyn_parallel(BH, chunk, [&](int64_t hb, int64_t he) {
    std::vector<float> Sbuf(NR * Lpad + 16);
    std::vector<uint16_t> Pbuf(NR * Lpad + 32, 0);
    float* S = Sbuf.data();
    uint16_t* Pb = Pbuf.data();
    const uint32_t* Pb32 = (const uint32_t*)Pb;
    for (int64_t bh = hb; bh < he; ++bh) {
      const int64_t b = bh / H, h = bh % H;
      const uint32_t* q = q0 + bh * L * P;
      const uint32_t* kp = kp0 + bh * P * Lpad;
      const uint32_t* vp = vp0 + bh * (Lpad / 2) * D;
      for (int64_t rg = 0; rg < nrg; ++rg) {
        const int64_t i0 = rg * NR;
        const int64_t nr = std::min<int64_t>(NR, L - i0);
        const uint32_t* qr[NR];
        for (int r = 0; r < NR; ++r) qr[r] = q + std::min<int64_t>(i0 + r, L - 1) * P;
        // ---- S = Q K^T  (NR rows x 64 keys per tile)
        __m512 mxv[NR];
        for (int r = 0; r < NR; ++r) mxv[r] = _mm512_set1_ps(-INFINITY);
        for (int64_t jt = 0; jt < Lpad; jt += 64) {
          __m512 acc[NR][4];
#pragma GCC unroll 6
          for (int r = 0; r < NR; ++r)
#pragma GCC unroll 4
            for (int v = 0; v < 4; ++v) acc[r][v] = _mm512_setzero_ps();
          const uint32_t* kr = kp + jt * P;  // tile-major: 64 keys x 32 pairs contiguous (8 KB)
#pragma GCC unroll 2
          for (int64_t p = 0; p < P; ++p) {
            const __m512bh b0 = (__m512bh)_mm512_loadu_si512(kr + p * 64);
            const __m512bh b1 = (__m512bh)_mm512_loadu_si512(kr + p * 64 + 16);
            const __m512bh b2 = (__m512bh)_mm512_loadu_si512(kr + p * 64 + 32);
            const __m512bh b3 = (__m512bh)_mm512_loadu_si512(kr + p * 64 + 48);
#pragma GCC unroll 6
            for (int r = 0; r < NR; ++r) {
              const __m512bh a = (__m512bh)_mm512_set1_epi32((int)qr[r][p]);
              acc[r][0] = _mm512_dpbf16_ps(acc[r][0], a, b0);
              acc[r][1] = _mm512_dpbf16_ps(acc[r][1], a, b1);
              acc[r][2] = _mm512_dpbf16_ps(acc[r][2], a, b2);
              acc[r][3] = _mm512_dpbf16_ps(acc[r][3], a, b3);
            }
          }
#pragma GCC unroll 6
          for (int r = 0; r < NR; ++r)
#pragma GCC unroll 4
            for (int v = 0; v < 4; ++v) _mm512_storeu_ps(S + r * Lpad + jt + 16 * v, acc[r][v]);
          // running row max over valid keys (fused, saves a pass over S)
          if (jt + 64 <= L) {
#pragma GCC unroll 6
            for (int r = 0; r < NR; ++r)
              mxv[r] = _mm512_max_ps(mxv[r], _mm512_max_ps(_mm512_max_ps(acc[r][0], acc[r][1]),
                                                           _mm512_max_ps(acc[r][2], acc[r][3])));
          } else {
            for (int v = 0; v < 4; ++v) {
              const int64_t nv = std::max<int64_t>(0, std::min<int64_t>(16, L - (jt + 16 * v)));
              const __mmask16 mk = (__mmask16)((1u << nv) - 1u);
              for (int r = 0; r < NR; ++r) mxv[r] = _mm512_mask_max_ps(mxv[r], mk, mxv[r], acc[r][v]);
            }
          }
        }
        // ---- softmax (exact, full row), unnormalised bf16 probabilities
        float inv[NR];
        for (int r = 0; r < nr; ++r) {
          const float* s = S + r * Lpad;
          const __m512 l2e = _mm512_set1_ps(1.44269504088896341f);
          const __m512 vml = _mm512_set1_ps(_mm512_reduce_max_ps(mxv[r]) * 1.44269504088896341f);
          __m512 sum = _mm512_setzero_ps();
          uint16_t* pr = Pb + r * Lpad;
          int64_t j = 0;
          for (; j + 16 <= L; j += 16) {  // e = 2^(s*log2e - m*log2e): one fmsub instead of sub+mul
            const __m512 e = exp2_512_fast(_mm512_fmsub_ps(_mm512_loadu_ps(s + j), l2e, vml));
            sum = _mm512_add_ps(sum, e);
            store_bf16x16(pr + j, e);
          }
          if (j < L) {
            const __mmask16 mk = (__mmask16)((1u << (L - j)) - 1u);
            const __m512 e = _mm512_maskz_mov_ps(mk, exp2_512_fast(_mm512_fmsub_ps(_mm512_loadu_ps(s + j), l2e, vml)));
            sum = _mm512_add_ps(sum, e);
            store_bf16x16(pr + j, e);
          }
          inv[r] = 1.0f / _mm512_reduce_add_ps(sum);
        }
        // ---- O = P V
        __m512 o[NR][4];
#pragma GCC unroll 6
        for (int r = 0; r < NR; ++r)
#pragma GCC unroll 4
          for (int v = 0; v < 4; ++v) o[r][v] = _mm512_setzero_ps();
        const int64_t pstride = Lpad / 2;
#pragma GCC unroll 2
        for (int64_t jp = 0; jp < npair; ++jp) {
          const uint32_t* vr = vp + jp * D;
          const __m512bh b0 = (__m512bh)_mm512_loadu_si512(vr);
          const __m512bh b1 = (__m512bh)_mm512_loadu_si512(vr + 16);
          const __m512bh b2 = (__m512bh)_mm512_loadu_si512(vr + 32);
          const __m512bh b3 = (__m512bh)_mm512_loadu_si512(vr + 48);
#pragma GCC unroll 6
          for (int r = 0; r < NR; ++r) {
            const __m512bh a = (__m512bh)_mm512_set1_epi32((int)Pb32[r * pstride + jp]);
            o[r][0] = _mm512_dpbf16_ps(o[r][0], a, b0);
            o[r][1] = _mm512_dpbf16_ps(o[r][1], a, b1);
            o[r][2] = _mm512_dpbf16_ps(o[r][2], a, b2);
            o[r][3] = _mm512_dpbf16_ps(o[r][3], a, b3);
          }
        }
        // ---- epilogue: gate * (1/sum), head merge, scatter to token order
        for (int r = 0; r < nr; ++r) {
          const int64_t i = i0 + r;
          const int64_t tok = mode == 0 ? i * F + b : b * F + i;
          const __m512 g = _mm512_set1_ps(g0[tok * H + h] * inv[r]);
          uint16_t* od = o0 + tok * (H * D) + h * D;
          store_bf16x16(od, _mm512_mul_ps(o[r][0], g));
          store_bf16x16(od + 16, _mm512_mul_ps(o[r][1], g));
          store_bf16x16(od + 32, _mm512_mul_ps(o[r][2], g));
          store_bf16x16(od + 48, _mm512_mul_ps(o[r][3], g));
        }
      }
    }
  });
}

// gate multiply + head merge for the torch-SDPA path: A [BH, L, 64] (any float dtype, contiguous) -> O [N, 512]
void gate_merge(Tensor A, Tensor G, Tensor O, int64_t mode, int64_t T, int64_t F) {
  constexpr int64_t H = 8, D = 64;
  const int64_t BH = A.size(0), L = A.size(1);
  TORCH_CHECK(A.is_contiguous() && A.scalar_type() == at::kBFloat16 && O.scalar_type() == at::kBFloat16);
  const uint16_t* a0 = (const uint16_t*)A.data_ptr();
  const float* g0 = G.data_ptr<float>();
  uint16_t* o0 = (uint16_t*)O.data_ptr();
  at::parallel_for(0, BH * L, 2048, [&](int64_t b_, int64_t e_) {
    for (int64_t idx = b_; idx < e_; ++idx) {
      const int64_t bh = idx / L, i = idx % L, b = bh / H, h = bh % H;
      const int64_t tok = mode == 0 ? i * F + b : b * F + i;
      const __m512 g = _mm512_set1_ps(g0[tok * H + h]);
      const uint16_t* src = a0 + idx * D;
      uint16_t* od = o0 + tok * (H * D) + h * D;
      for (int v = 0; v < 4; ++v) store_bf16x16(od + 16 * v, _mm512_mul_ps(load_bf16x16(src + 16 * v), g));
    }
  });
}

// ================================================================ fused GEMM kernels ("megakernel" layer path)
// Weights are pre-packed in VNNI tiles: Wp[tile][K/2][64] int32, Wp[t][kp][c] = (W[t*64+c][2kp], W[t*64+c][2kp+1]).
// Microkernel: 6 rows x (16*NV) columns, acc += A_pair (broadcast) * B_pairs via vdpbf16ps, fp32 accumulation.
template <int NV>
static inline void mk6(const uint32_t* const* a, const uint32_t* bt, int64_t KP, __m512 (&acc)[6][4]) {
#pragma GCC unroll 6
  for (int r = 0; r < 6; ++r)
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) acc[r][v] = _mm512_setzero_ps();
#pragma GCC unroll 2
  for (int64_t kp = 0; kp < KP; ++kp) {
    __m512bh b[NV];
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) b[v] = (__m512bh)_mm512_loadu_si512(bt + kp * 64 + 16 * v);
#pragma GCC unroll 6
    for (int r = 0; r < 6; ++r) {
      const __m512bh av = (__m512bh)_mm512_set1_epi32((int)a[r][kp]);
#pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) acc[r][v] = _mm512_dpbf16_ps(acc[r][v], av, b[v]);
    }
  }
}

static inline __m512i pack_pairs_bf16(__m512 lo, __m512 hi) {  // u32 lanes = bf16(lo) | bf16(hi) << 16
  const __m512i r0 = _mm512_cvtepu16_epi32((__m256i)_mm512_cvtneps_pbh(lo));
  const __m512i r1 = _mm512_cvtepu16_epi32((__m256i)_mm512_cvtneps_pbh(hi));
  return _mm512_or_si512(r0, _mm512_slli_epi32(r1, 16));
}

// K1: xn [N,256] bf16 -> qkv+gate GEMM -> rope (fp32 accumulators) -> Q [BH,L,64] bf16, Kp [BH,32,Lpad] u32,
//     Vp [BH,Lpad/2,64] u32, G [N,8] = sigmoid(gate + bias). Wp: 25 tiles (q 0-7, k 8-15, v 16-23, gate 24).
//     cdup/sdup: [L,64] fp32, cdup[j][2p]=cdup[j][2p+1]=cos, sdup[j][2p]=-sin, sdup[j][2p+1]=+sin.
void qkv_rope_gemm(Tensor xn, Tensor Wp, Tensor gbias, Tensor cdup, Tensor sdup, int64_t mode, int64_t T, int64_t F,
                   Tensor Q, Tensor Kp, Tensor Vp, Tensor G) {
  constexpr int64_t H = 8, D = 64, P = 32, KP = 128, RB = 48;
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && xn.size(1) == 256);
  TORCH_CHECK(Wp.size(0) == 25 && Wp.size(1) == KP && Wp.size(2) == 64 && Wp.is_contiguous());
  const int64_t B = mode == 0 ? F : T, L = mode == 0 ? T : F, Lpad = Kp.size(2);
  TORCH_CHECK(Q.size(0) == B * H && Q.size(1) == L && Kp.size(0) == B * H && Vp.size(1) == Lpad / 2);
  TORCH_CHECK(cdup.size(0) >= L && cdup.size(1) == 64);
  const uint32_t* x0 = (const uint32_t*)xn.data_ptr();
  const uint32_t* w0 = (const uint32_t*)Wp.data_ptr();
  const float* gb = gbias.data_ptr<float>();
  const float* cd = cdup.data_ptr<float>();
  const float* sd = sdup.data_ptr<float>();
  uint16_t* q0 = (uint16_t*)Q.data_ptr();
  uint32_t* k0 = (uint32_t*)Kp.data_ptr();
  uint32_t* v0 = (uint32_t*)Vp.data_ptr();
  float* g0 = G.data_ptr<float>();
  const int64_t nrb = (L + RB - 1) / RB;
  dyn_parallel(B * nrb, 1, [&](int64_t tb, int64_t te) {
    for (int64_t task = tb; task < te; ++task) {
      const int64_t b = task / nrb, jb = (task % nrb) * RB;
      const int64_t nrows = std::min(RB, L - jb);
      const int64_t ngr = (nrows + 5) / 6;
      for (int64_t ct = 0; ct < 25; ++ct) {
        const uint32_t* bt = w0 + ct * KP * 64;
        for (int64_t g = 0; g < ngr; ++g) {
          const int64_t j0 = jb + g * 6;
          const int nr = (int)std::min<int64_t>(6, L - j0);
          const uint32_t* a[6];
          int64_t tok[6];
          for (int r = 0; r < 6; ++r) {
            const int64_t j = j0 + std::min(r, nr - 1);
            tok[r] = mode == 0 ? j * F + b : b * F + j;
            a[r] = x0 + tok[r] * KP;
          }
          __m512 acc[6][4];
          if (ct == 24) {
            mk6<1>(a, bt, KP, acc);
            const __m512 bias = _mm512_maskz_loadu_ps(0xFF, gb);
            for (int r = 0; r < nr; ++r) {
              const __m512 z = _mm512_add_ps(acc[r][0], bias);
              const __m512 sg = _mm512_div_ps(_mm512_set1_ps(1.0f),
                                              _mm512_add_ps(_mm512_set1_ps(1.0f), exp512(_mm512_sub_ps(_mm512_setzero_ps(), z))));
              _mm512_mask_storeu_ps(g0 + tok[r] * H, 0xFF, sg);
            }
            continue;
          }
          mk6<4>(a, bt, KP, acc);
          const int64_t h = ct % 8, kind = ct / 8;  // 0 q, 1 k, 2 v
          const int64_t bh = b * H + h;
          if (kind < 2) {
            alignas(64) uint32_t kt[6][P];
            for (int r = 0; r < nr; ++r) {
              const int64_t j = j0 + r;
              const float* c = cd + j * 64;
              const float* s = sd + j * 64;
              __m512i pr[2];
              __m512 o[4];
              for (int v = 0; v < 4; ++v) {
                const __m512 x = acc[r][v];
                o[v] = _mm512_fmadd_ps(x, _mm512_loadu_ps(c + 16 * v),
                                       _mm512_mul_ps(_mm512_permute_ps(x, 0xB1), _mm512_loadu_ps(s + 16 * v)));
              }
              // bf16 row (64 values = 32 pairs)
              pr[0] = _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(o[0]));
              pr[0] = _mm512_inserti64x4(pr[0], (__m256i)_mm512_cvtneps_pbh(o[1]), 1);
              pr[1] = _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(o[2]));
              pr[1] = _mm512_inserti64x4(pr[1], (__m256i)_mm512_cvtneps_pbh(o[3]), 1);
              if (kind == 0) {
                uint16_t* qd = q0 + (bh * L + j) * D;
                _mm512_storeu_si512(qd, pr[0]);
                _mm512_storeu_si512(qd + 32, pr[1]);
              } else {
                _mm512_store_si512(kt[r], pr[0]);
                _mm512_store_si512(kt[r] + 16, pr[1]);
              }
            }
            if (kind == 1) {
              uint32_t* kb = k0 + bh * P * Lpad;
              for (int r = 0; r < nr; ++r) {
                const int64_t j = j0 + r;
                uint32_t* kp = kb + (j >> 6) * (P * 64) + (j & 63);
                for (int64_t p = 0; p < P; ++p) kp[p * 64] = kt[r][p];
              }
            }
          } else {
            uint32_t* vp = v0 + (bh * (Lpad / 2) + j0 / 2) * D;
            for (int r = 0; r < nr; r += 2) {
              for (int v = 0; v < 4; ++v) {
                const __m512 hi = (r + 1 < nr) ? acc[r + 1][v] : _mm512_setzero_ps();
                _mm512_storeu_si512(vp + (r / 2) * D + 16 * v, pack_pairs_bf16(acc[r][v], hi));
              }
            }
          }
        }
      }
    }
  });
}

// K3/K5: X [N,256] fp32 += A [N,K] bf16 @ W^T (+bias) ; xn = X / max(||X||, eps) as bf16.  Wp: 4 tiles [K/2][64]
void gemm_res_norm(Tensor A, Tensor Wp, const std::optional<Tensor>& bias, Tensor X, Tensor xn, double eps) {
  const int64_t N = A.size(0), K = A.size(1), KP = K / 2;
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && A.is_contiguous() && K % 2 == 0);
  TORCH_CHECK(Wp.size(0) == 4 && Wp.size(1) == KP && Wp.size(2) == 64);
  TORCH_CHECK(X.scalar_type() == at::kFloat && X.size(0) == N && X.size(1) == 256 && X.is_contiguous());
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous());
  const uint32_t* a0 = (const uint32_t*)A.data_ptr();
  const uint32_t* w0 = (const uint32_t*)Wp.data_ptr();
  const bool hb = bias.has_value() && bias->defined();
  const float* bs = hb ? bias->data_ptr<float>() : nullptr;
  float* x0 = X.data_ptr<float>();
  uint16_t* n0 = (uint16_t*)xn.data_ptr();
  const float feps = (float)eps;
  const int64_t ng = (N + 5) / 6;
  dyn_parallel(ng, 8, [&](int64_t gb_, int64_t ge_) {
    for (int64_t g = gb_; g < ge_; ++g) {
      const int64_t i0 = g * 6;
      const int nr = (int)std::min<int64_t>(6, N - i0);
      const uint32_t* a[6];
      for (int r = 0; r < 6; ++r) a[r] = a0 + (i0 + std::min(r, nr - 1)) * KP;
      __m512 ss[6];
      for (int r = 0; r < 6; ++r) ss[r] = _mm512_setzero_ps();
      for (int64_t ct = 0; ct < 4; ++ct) {
        __m512 acc[6][4];
        mk6<4>(a, w0 + ct * KP * 64, KP, acc);
        for (int r = 0; r < nr; ++r) {
          float* xr = x0 + (i0 + r) * 256 + ct * 64;
          for (int v = 0; v < 4; ++v) {
            __m512 y = _mm512_add_ps(acc[r][v], _mm512_loadu_ps(xr + 16 * v));
            if (hb) y = _mm512_add_ps(y, _mm512_loadu_ps(bs + ct * 64 + 16 * v));
            _mm512_storeu_ps(xr + 16 * v, y);
            ss[r] = _mm512_fmadd_ps(y, y, ss[r]);
          }
        }
      }
      for (int r = 0; r < nr; ++r) {
        const float* xr = x0 + (i0 + r) * 256;
        const __m512 vi = _mm512_set1_ps(1.0f / std::max(std::sqrt(_mm512_reduce_add_ps(ss[r])), feps));
        uint16_t* nd = n0 + (i0 + r) * 256;
        for (int v = 0; v < 16; ++v) store_bf16x16(nd + 16 * v, _mm512_mul_ps(_mm512_loadu_ps(xr + 16 * v), vi));
      }
    }
  });
}

// erf-GELU (Abramowitz-Stegun 7.1.26 on |x|/sqrt2, odd extension; abs. err of erf ~1e-5, far below the bf16 output step)
static inline __m512 gelu512(__m512 x) {
  const __m512 z = _mm512_mul_ps(x, _mm512_set1_ps(0.70710678118654752f));
  const __m512 az = _mm512_abs_ps(z);
  const __m512 den = _mm512_fmadd_ps(az, _mm512_set1_ps(0.3275911f), _mm512_set1_ps(1.0f));
  __m512 t = _mm512_rcp14_ps(den);                                              // 2^-14 rel. error
  t = _mm512_mul_ps(t, _mm512_fnmadd_ps(den, t, _mm512_set1_ps(2.0f)));         // one Newton step (no vdivps)
  __m512 p = _mm512_set1_ps(1.061405429f);
  p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(-1.453152027f));
  p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(1.421413741f));
  p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(-0.284496736f));
  p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(0.254829592f));
  p = _mm512_mul_ps(p, t);
  const __m512 e = exp512_fast(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(az, az)));  // result is rounded to bf16
  __m512 erf_abs = _mm512_fnmadd_ps(p, e, _mm512_set1_ps(1.0f));  // 1 - p*e
  // restore sign of z
  const __m512 erfz = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(erf_abs),
                        _mm512_and_si512(_mm512_castps_si512(z), _mm512_set1_epi32((int)0x80000000u))));
  return _mm512_mul_ps(_mm512_mul_ps(x, _mm512_set1_ps(0.5f)), _mm512_add_ps(_mm512_set1_ps(1.0f), erfz));
}

// K4: Hout [N,1024] bf16 = gelu(xn [N,256] @ W1^T + b1).  Wp: 16 tiles [128][64]
void gemm_gelu(Tensor xn, Tensor Wp, Tensor bias, Tensor Hout) {
  const int64_t N = xn.size(0), KP = xn.size(1) / 2, NT = Wp.size(0);
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && Wp.size(1) == KP);
  TORCH_CHECK(Hout.size(1) == NT * 64 && Hout.scalar_type() == at::kBFloat16);
  const uint32_t* a0 = (const uint32_t*)xn.data_ptr();
  const uint32_t* w0 = (const uint32_t*)Wp.data_ptr();
  const float* bs = bias.data_ptr<float>();
  uint16_t* h0 = (uint16_t*)Hout.data_ptr();
  const int64_t ldh = NT * 64, RB = 48;
  const int64_t nrb = (N + RB - 1) / RB;
  dyn_parallel(nrb, 1, [&](int64_t tb, int64_t te) {
    for (int64_t task = tb; task < te; ++task) {
      const int64_t ib = task * RB;
      const int64_t ngr = (std::min(RB, N - ib) + 5) / 6;
      for (int64_t ct = 0; ct < NT; ++ct) {
        const uint32_t* bt = w0 + ct * KP * 64;
        for (int64_t g = 0; g < ngr; ++g) {
          const int64_t i0 = ib + g * 6;
          const int nr = (int)std::min<int64_t>(6, N - i0);
          const uint32_t* a[6];
          for (int r = 0; r < 6; ++r) a[r] = a0 + (i0 + std::min(r, nr - 1)) * KP;
          __m512 acc[6][4];
          mk6<4>(a, bt, KP, acc);
          for (int r = 0; r < nr; ++r) {
            uint16_t* hd = h0 + (i0 + r) * ldh + ct * 64;
            for (int v = 0; v < 4; ++v)
              store_bf16x16(hd + 16 * v, gelu512(_mm512_add_ps(acc[r][v], _mm512_loadu_ps(bs + ct * 64 + 16 * v))));
          }
        }
      }
    }
  });
}

TORCH_LIBRARY(turbo_roformer, m) {
  m.def("add_rmsnorm", &add_rmsnorm);
  m.def("rope_pack", &rope_pack);
  m.def("attention", &attention);
  m.def("gate_merge", &gate_merge);
  m.def("qkv_rope_gemm", &qkv_rope_gemm);
  m.def("gemm_res_norm", &gemm_res_norm);
  m.def("gemm_gelu", &gemm_gelu);
}
