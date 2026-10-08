"""Restructured, inference-only BS-RoFormer forward for CPUs (same weights, same math).

``FastBSRoformer(model)`` wraps a loaded :class:`turbo_roformer.model.BSRoformer` and replaces its forward:

* RMSNorm ``gamma * sqrt(dim)`` is folded into the following Linear, the attention scale (1/8, exact in fp32
  and bf16) into ``W_q``; every norm becomes a plain L2 normalisation.
* ``to_qkv`` and ``to_gates`` are fused into one GEMM (256 -> 1544).
* Rotary embeddings use precomputed fp32 cos/sin tables (no cat/neg/rotate_half).
* No rearrange/pack copies between the time and frequency transformers: the token layout stays (t, f, d) and
  the axial grouping is done by index arithmetic (C++ kernels) or strided views (torch path).
* The 62 band MLPs of the mask estimator are batched into a handful of grouped ``bmm`` calls.
* Optional AVX512-BF16 C++ kernels (``glue="cpp"``): bf16 GEMMs with fp32 accumulation, a custom exact-softmax
  attention kernel, and fused GEMM epilogues (rotary + VNNI packing, residual add + next RMSNorm, GELU).
  The residual stream, STFT, mask and iSTFT stay fp32.

Configurations (used by the benchmark ladder and the backends):

=========================  ==================================================================
``dtype=fp32, glue=torch``  exact restructured path (>= 100 dB vs the reference forward)
``dtype=bf16, glue=torch``  same, bf16 GEMMs/SDPA (portable, no compiler needed)
``glue=cpp, attn=sdpa``     C++ norm/rotary/packing glue, torch bf16 SDPA
``glue=cpp, attn=cpp``      + custom attention kernel
``fused="best"``            + fused qkv/rotary GEMM, fused out-proj+residual+norm, fused FF2+residual+norm
=========================  ==================================================================

Instances keep preallocated work buffers: one ``forward`` at a time per instance.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from . import _kernels

__all__ = ["FastBSRoformer"]


def _pack_w(W: torch.Tensor) -> torch.Tensor:
    """(Nout, K) float -> VNNI tiles int32 [ceil(Nout/64), K/2, 64]; tile[t][kp][c] = bf16 pair W[t*64+c, 2kp:2kp+2]."""
    n_out, k = W.shape
    nt = (n_out + 63) // 64
    Wb = torch.zeros(nt * 64, k, dtype=torch.bfloat16)
    Wb[:n_out] = W.to(torch.bfloat16)
    return Wb.view(torch.int32).view(nt, 64, k // 2).transpose(1, 2).contiguous()


def _mkldnn_gelu_linear_available() -> bool:
    try:
        x = torch.zeros(2, 32, dtype=torch.bfloat16)
        w = torch.zeros(32, 32, dtype=torch.bfloat16)
        torch.ops.mkldnn._linear_pointwise(x, w, None, "gelu", [], "none")
        return True
    except Exception:
        return False


class _Timer:
    """Optional per-section wall clock profile (``fast.timer.enabled = True``)."""

    def __init__(self):
        self.enabled = False
        self.t: dict[str, float] = defaultdict(float)

    def __call__(self, key: str):
        tm = self

        class _Section:
            def __enter__(s):
                s.t0 = time.perf_counter() if tm.enabled else 0.0

            def __exit__(s, *exc):
                if tm.enabled:
                    tm.t[key] += time.perf_counter() - s.t0

        return _Section()


class FastBSRoformer(torch.nn.Module):
    H, DH = 8, 64

    def __init__(
        self,
        model,
        dtype=torch.bfloat16,
        glue="cpp",
        attn="cpp",
        fused="best",
        gelu_fused=True,
        mask_dtype=torch.float32,
    ):
        super().__init__()
        if glue not in ("torch", "cpp") or attn not in ("sdpa", "cpp"):
            raise ValueError("glue must be torch|cpp, attn must be sdpa|cpp")
        if attn == "cpp" and glue != "cpp":
            raise ValueError("attn='cpp' requires glue='cpp'")
        if glue == "cpp":
            if dtype != torch.bfloat16:
                raise ValueError("the C++ kernels are bf16 only")
            _kernels.load()
        m = model
        if not (m.num_stems == 1 and m.stereo and m.audio_channels == 2):
            raise NotImplementedError("only single-stem stereo BS-RoFormer models are supported")
        if (m.heads, m.dim_head) != (self.H, self.DH):
            raise NotImplementedError(f"only heads=8, dim_head=64 is supported (got {m.heads}, {m.dim_head})")
        self.model = m
        self.dtype, self.glue, self.attn_impl = dtype, glue, attn
        self.gelu_fused = bool(gelu_fused and dtype == torch.bfloat16 and _mkldnn_gelu_linear_available())
        self.mask_dtype = mask_dtype
        dim = m.final_norm.gamma.shape[0]
        self.dim = dim
        if fused in ("best", "all"):
            fused = ("qkv", "out", "ff2") if fused == "best" else ("qkv", "out", "ff")
        self.fused = set(fused or ())
        if self.fused and not (glue == "cpp" and attn == "cpp"):
            raise ValueError("fused GEMM kernels require glue='cpp' and attn='cpp'")
        if self.fused and dim != 256:
            self.fused = set()  # the fused epilogue kernels are specialised for dim 256
        self._bufs: dict[str, torch.Tensor] = {}
        self.timer = _Timer()
        self.layers = []
        with torch.no_grad():
            for blk in m.layers:
                if len(blk) != 2:
                    raise NotImplementedError("linear transformers are not supported")
                for mode, tr in enumerate(blk):  # 0 = time, 1 = freq
                    if len(tr.layers) != 1 or not isinstance(tr.norm, torch.nn.Identity):
                        raise NotImplementedError("only time/freq transformer depth 1 without output norm is supported")
                    attn_m, ff = tr.layers[0]
                    self.layers.append(self._prep(attn_m, ff, mode))
            t_rot = m.layers[0][0].layers[0][0].rotary_embed
            f_rot = m.layers[0][1].layers[0][0].rotary_embed
            self.rot = {0: t_rot.freqs.detach().float().clone(), 1: f_rot.freqs.detach().float().clone()}
            self._tab_cache: dict = {}
            self.final_g = (m.final_norm.gamma * m.final_norm.scale).float().clone()
            # mask estimator: consecutive bands with equal width share one batched MLP
            me = m.mask_estimators[0]
            dims = list(me.dim_inputs)
            groups, i, off = [], 0, 0
            while i < len(dims):
                j = i
                while j < len(dims) and dims[j] == dims[i]:
                    j += 1
                mlps = [me.to_freqs[k][0] for k in range(i, j)]  # Sequential(Linear, Tanh, Linear)
                for mlp in mlps:
                    if len(mlp) != 3 or not isinstance(mlp[1], torch.nn.Tanh):
                        raise NotImplementedError("only mask_estimator_depth=2 is supported")
                W1 = torch.stack([mm[0].weight.t() for mm in mlps]).contiguous().to(mask_dtype)
                b1 = torch.stack([mm[0].bias for mm in mlps]).unsqueeze(1).contiguous().to(mask_dtype)
                W2 = torch.stack([mm[2].weight.t() for mm in mlps]).contiguous().to(mask_dtype)
                b2 = torch.stack([mm[2].bias for mm in mlps]).unsqueeze(1).contiguous().to(mask_dtype)
                groups.append(dict(i0=i, i1=j, din=dims[i], off=off, W1=W1, b1=b1, W2=W2, b2=b2))
                off += dims[i] * (j - i)
                i = j
            self.mask_groups = groups
            self.mask_width = off

    # ------------------------------------------------------------------ weight preparation
    def _prep(self, a, ff, mode):
        H, DH, dt = self.H, self.DH, self.dtype
        ga = (a.norm.gamma * a.norm.scale).float()
        Wqkv = a.to_qkv.weight.float() * ga[None, :]
        Wqkv[: H * DH] *= a.scale  # 1/8: exact in fp32 and bf16
        Wg = a.to_gates.weight.float() * ga[None, :]
        W = torch.cat([Wqkv, Wg], 0)
        b = torch.cat([torch.zeros(3 * H * DH), a.to_gates.bias.float()])
        nrm, l1, l2 = ff.net[0], ff.net[1], ff.net[4]
        gf = (nrm.gamma * nrm.scale).float()
        P = dict(
            mode=mode,
            W=W.to(dt).contiguous(),
            b=b.to(dt).contiguous(),
            Wo=a.to_out[0].weight.float().to(dt).contiguous(),
            W1=(l1.weight.float() * gf[None, :]).to(dt).contiguous(),
            b1=l1.bias.float().to(dt).contiguous(),
            W2=l2.weight.float().to(dt).contiguous(),
            b2=l2.bias.float().to(dt).contiguous(),
        )
        if self.glue == "cpp":
            P["Wp"] = _pack_w(W)  # 25 tiles: q 0-7, k 8-15, v 16-23, gates 24
            P["bg"] = a.to_gates.bias.float().contiguous()
            P["Wop"] = _pack_w(a.to_out[0].weight.float())
            P["W1p"] = _pack_w(l1.weight.float() * gf[None, :])
            P["b1f"] = l1.bias.float().contiguous()
            P["W2p"] = _pack_w(l2.weight.float())
            P["b2f"] = l2.bias.float().contiguous()
        return P

    def _tables(self, mode, L):
        key = (mode, L)
        if key not in self._tab_cache:
            pos = torch.arange(L, dtype=torch.float32)
            ang = pos[:, None] * self.rot[mode][None, :]  # (L, 32), fp32 exactly like the reference
            self._tab_cache[key] = (ang.cos().contiguous(), ang.sin().contiguous())
        return self._tab_cache[key]

    def _dup_tables(self, mode, L):
        key = ("dup", mode, L)
        if key not in self._tab_cache:
            c, s = self._tables(mode, L)
            cd = c.repeat_interleave(2, dim=1).contiguous()
            sd = torch.stack((-s, s), -1).flatten(1).contiguous()
            self._tab_cache[key] = (cd, sd)
        return self._tab_cache[key]

    def _buf(self, name, shape, dtype, zero=False):
        b = self._bufs.get(name)
        if b is None or tuple(b.shape) != tuple(shape) or b.dtype != dtype:
            b = (torch.zeros if zero else torch.empty)(shape, dtype=dtype)
            self._bufs[name] = b
        return b

    def release_buffers(self):
        """Free the cached work buffers (they are re-created on the next forward)."""
        self._bufs.clear()

    # ------------------------------------------------------------------ transformer (torch glue)
    @staticmethod
    def _rope_t(x, c, s):
        # x (..., L, 64), c/s (L, 32), interleaved pairs; fp32 math
        xf = x.float().unflatten(-1, (32, 2))
        x0, x1 = xf[..., 0], xf[..., 1]
        return torch.stack((x0 * c - x1 * s, x1 * c + x0 * s), -1).flatten(-2).to(x.dtype)

    def _attn_torch(self, X, P, T, Fq):
        H, DH = self.H, self.DH
        mode = P["mode"]
        xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        qkvg = F.linear(xn.to(self.dtype), P["W"], P["b"])  # (N, 1544)
        qkv = qkvg[:, : 3 * H * DH].view(T, Fq, 3, H, DH)
        g = qkvg[:, 3 * H * DH :].float().sigmoid()  # (N, H)
        if mode == 0:
            qkv = qkv.permute(2, 1, 3, 0, 4)  # (3, F, H, T, D)
            L = T
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, T, H, F, D)
            L = Fq
        c, s = self._tables(mode, L)
        q = self._rope_t(qkv[0], c, s)
        k = self._rope_t(qkv[1], c, s)
        v = qkv[2]
        o = F.scaled_dot_product_attention(q, k, v, scale=1.0)  # scale folded into W_q
        o = o.permute(2, 0, 1, 3) if mode == 0 else o.permute(0, 2, 1, 3)  # (T, F, H, D)
        o = (o * g.view(T, Fq, H, 1).to(o.dtype)).reshape(T * Fq, H * DH)
        return F.linear(o, P["Wo"])

    def _ff_torch(self, X, P):
        xn = (X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(self.dtype)
        if self.gelu_fused:
            h = torch.ops.mkldnn._linear_pointwise(xn, P["W1"], P["b1"], "gelu", [], "none")
        else:
            h = F.gelu(F.linear(xn, P["W1"], P["b1"]))
        return F.linear(h, P["W2"], P["b2"])

    def _transformer_torch(self, X, T, Fq):
        tm = self.timer
        for P in self.layers:
            with tm("attn_block"):
                X = X + self._attn_torch(X, P, T, Fq).float()
            with tm("ff_block"):
                X = X + self._ff_torch(X, P).float()
        return X

    # ------------------------------------------------------------------ transformer (C++ glue)
    def _transformer_cpp(self, X, T, Fq):
        ops = torch.ops.turbo_roformer
        tm = self.timer
        H, DH = self.H, self.DH
        N = X.shape[0]
        fu = self.fused
        main_thread = threading.current_thread() is threading.main_thread()
        bf = torch.bfloat16
        xn = self._buf("xn", (N, self.dim), bf)
        O = self._buf("O", (N, H * DH), bf)
        with tm("norm"):
            ops.add_rmsnorm(X, None, xn, 1e-12)
        for P in self.layers:
            mode = P["mode"]
            L = T if mode == 0 else Fq
            B = Fq if mode == 0 else T
            c, s = self._tables(mode, L)
            packed = self.attn_impl == "cpp"
            Lpad = ((L + 63) // 64) * 64
            if "qkv" in fu:
                cd, sd = self._dup_tables(mode, L)
                Q = self._buf(f"Q{mode}", (B * H, L, DH), bf)
                K = self._buf(f"K{mode}", (B * H, 32, Lpad), torch.int32, zero=True)  # padding stays zero
                V = self._buf(f"V{mode}", (B * H, Lpad // 2, DH), torch.int32, zero=True)
                G = self._buf("G", (N, H), torch.float32)
                with tm("qkv_rope_gemm"):
                    ops.qkv_rope_gemm(xn, P["Wp"], P["bg"], cd, sd, mode, T, Fq, Q, K, V, G)
            else:
                with tm("gemm_qkvg"):
                    qkvg = F.linear(xn, P["W"], P["b"])
                with tm("rope_pack"):
                    Q, K, V, G = ops.rope_pack(qkvg, c, s, mode, T, Fq, Lpad, packed)
                del qkvg
            with tm("attention_t" if mode == 0 else "attention_f"):
                if packed:
                    ops.attention(Q, K, V, G, O, mode, T, Fq)
                else:
                    A = F.scaled_dot_product_attention(
                        Q.view(-1, H, L, DH), K.view(-1, H, L, DH), V.view(-1, H, L, DH), scale=1.0
                    )
                    ops.gate_merge(A.reshape(-1, L, DH).contiguous(), G, O, mode, T, Fq)
            del Q, K, V
            if "out" in fu:
                with tm("gemm_out_res_norm"):
                    ops.gemm_res_norm(O, P["Wop"], None, X, xn, 1e-12)
            else:
                with tm("gemm_out"):
                    y = F.linear(O, P["Wo"])
                with tm("norm"):
                    ops.add_rmsnorm(X, y, xn, 1e-12)
            if ("ff2" in fu and not main_thread) or "ff" in fu:
                # Off the main thread glibc serves blocks > 64 MB of non-main arenas with fresh mmaps regardless of
                # mallopt, so avoid the per-layer 216 MB oneDNN output and use the allocation-free GELU kernel.
                Hb = self._buf("H", (N, P["W1p"].shape[0] * 64), bf)
                with tm("gemm_ff1_gelu"):
                    ops.gemm_gelu(xn, P["W1p"], P["b1f"], Hb)
                with tm("gemm_ff2_res_norm"):
                    ops.gemm_res_norm(Hb, P["W2p"], P["b2f"], X, xn, 1e-12)
            elif "ff2" in fu:  # oneDNN GEMM+GELU for ff1 (faster than the custom epilogue), fused ff2+residual+norm
                with tm("gemm_ff1"):
                    h = self._ff1(xn, P)
                with tm("gemm_ff2_res_norm"):
                    ops.gemm_res_norm(h, P["W2p"], P["b2f"], X, xn, 1e-12)
                del h
            else:
                with tm("gemm_ff1"):
                    h = self._ff1(xn, P)
                with tm("gemm_ff2"):
                    y = F.linear(h, P["W2"], P["b2"])
                del h
                with tm("norm"):
                    ops.add_rmsnorm(X, y, xn, 1e-12)
        return X

    def _ff1(self, xn, P):
        if self.gelu_fused:
            return torch.ops.mkldnn._linear_pointwise(xn, P["W1"], P["b1"], "gelu", [], "none")
        return F.gelu(F.linear(xn, P["W1"], P["b1"]))

    # ------------------------------------------------------------------ mask estimator
    def _mask(self, Xn, T, Fq):
        mask = torch.empty(T, self.mask_width, dtype=torch.float32)
        Xt = Xn.to(self.mask_dtype).transpose(0, 1)  # (F, T, d) view
        for g in self.mask_groups:
            xb = Xt[g["i0"] : g["i1"]]
            h = torch.baddbmm(g["b1"], xb, g["W1"]).tanh_()
            o = torch.baddbmm(g["b2"], h, g["W2"])  # (nb, T, 2*din)
            din = g["din"]
            o = (o[..., :din] * o[..., din:].sigmoid()).float()  # GLU
            nb = g["i1"] - g["i0"]
            mask[:, g["off"] : g["off"] + nb * din] = o.transpose(0, 1).reshape(T, nb * din)
        return mask

    # ------------------------------------------------------------------ full forward
    @torch.inference_mode()
    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """raw_audio (1, 2, samples) float32 -> (1, 2, samples') with samples' = hop * (samples // hop)."""
        m, tm = self.model, self.timer
        if raw_audio.ndim != 3 or raw_audio.shape[0] != 1 or raw_audio.shape[1] != 2:
            raise ValueError(f"expected (1, 2, samples), got {tuple(raw_audio.shape)}")
        raw_audio = raw_audio.float()
        with tm("stft"):
            x = raw_audio.reshape(2, -1)
            win = m.stft_window_fn()
            st = torch.view_as_real(torch.stft(x, **m.stft_kwargs, window=win, return_complex=True))  # (2, F, T, 2)
            Fbins, T = st.shape[1], st.shape[2]
            st = st.permute(1, 0, 2, 3).reshape(1, Fbins * 2, T, 2)  # b (f s) t c
            xin = st.permute(0, 2, 1, 3).reshape(1, T, Fbins * 2 * 2)  # b t (f c)
        with tm("band_split"):
            X = m.band_split(xin)  # (1, T, bands, d) fp32
            Fq = X.shape[2]
            X = X.reshape(T * Fq, self.dim).contiguous()
        if self.glue == "cpp":
            X = self._transformer_cpp(X, T, Fq)
        else:
            X = self._transformer_torch(X, T, Fq)
        with tm("mask"):
            Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.final_g
            mask = self._mask(Xn.view(T, Fq, self.dim), T, Fq)  # (T, bins * 2 * 2)
        with tm("istft"):
            mask = mask.view(T, Fbins * 2, 2).transpose(0, 1)  # (f s, t, c)
            stc = torch.view_as_complex(st[0].contiguous())
            mc = torch.view_as_complex(mask.contiguous())
            spec = (stc * mc).view(Fbins, 2, T).transpose(0, 1)  # (s, f, t)
            out = torch.istft(spec, **m.stft_kwargs, window=win, return_complex=False)
        return out.unsqueeze(0)
