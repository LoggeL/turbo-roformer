"""Reference BS-RoFormer module (inference), parameter-compatible with existing checkpoints.

This is a dependency-free port of the BS-RoFormer implementation used by python-audio-separator
(``audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py``), which in turn derives from
Phil Wang's (lucidrains) BS-RoFormer (https://github.com/lucidrains/BS-RoFormer). Both are MIT licensed;
see NOTICE. The port keeps the module tree and therefore the ``state_dict`` keys identical, so the original
checkpoints load with ``strict=True``. ``einops``/``beartype``/``rotary-embedding-torch`` are replaced by plain
torch ops that compute the same thing in the same order (this is the "reference forward" that the optimized
paths in :mod:`turbo_roformer.fast` are tested against). Training-only code (losses, linear attention) is
omitted; ``linear_transformer_depth`` must be 0.

Paper: Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, Yun-Ning Hung, "Music Source Separation with
Band-Split RoPE Transformer", ICASSP 2024 (arXiv:2309.02612).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_FREQS_PER_BANDS: tuple[int, ...] = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    """Interleaved-pair rotary embedding ("lang" frequencies, theta 10000), as rotary-embedding-torch 0.6."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def angles(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """fp32 angles (seq_len, dim), each frequency repeated for its (even, odd) pair."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        ang = torch.einsum("i,f->if", pos, self.freqs.float())
        return torch.repeat_interleave(ang, 2, dim=-1)

    def rotate(self, t: torch.Tensor) -> torch.Tensor:
        """t (..., seq, dim) -> rotated t (rotation applied in fp32, result cast back to t.dtype)."""
        ang = self.angles(t.shape[-2], t.device)
        tf = t.float()
        x = tf.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(-1)
        rot_half = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (tf * ang.cos() + rot_half * ang.sin()).to(t.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: RotaryEmbedding | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = self.norm(x)
        # "b n (qkv h d) -> qkv b h n d"
        q, k, v = self.to_qkv(x).view(b, n, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate(q)
            k = self.rotary_embed.rotate(k)
        out = F.scaled_dot_product_attention(q, k, v)  # default scale = dim_head ** -0.5
        gates = self.to_gates(x)  # (b, n, h)
        out = out * gates.permute(0, 2, 1).unsqueeze(-1).sigmoid()
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)  # "b h n d -> b n (h d)"
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: RotaryEmbedding | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_embed=rotary_embed),
                    FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                ]
            )
            for _ in range(depth)
        )
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList(nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = x.split(self.dim_inputs, dim=-1)
        return torch.stack([net(p) for p, net in zip(parts, self.to_features)], dim=-2)


def _mlp(dim_in: int, dim_out: int, dim_hidden: int, depth: int) -> nn.Sequential:
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], depth: int, mlp_expansion_factor: int = 4):
        super().__init__()
        self.dim_inputs = dim_inputs
        hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList(
            nn.Sequential(_mlp(dim, d * 2, hidden, depth), nn.GLU(dim=-1)) for d in dim_inputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([mlp(band) for band, mlp in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)


class BSRoformer(nn.Module):
    """Band-Split RoPE Transformer (inference only). ``forward(raw_audio (b, s, t)) -> (b, s, t)`` for one stem."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        linear_transformer_depth: int = 0,
        freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        mlp_expansion_factor: int = 4,
        dim_freqs_in: int = 1025,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        mask_estimator_depth: int = 2,
        **_ignored: Any,
    ):
        super().__init__()
        if linear_transformer_depth:
            raise NotImplementedError("linear_transformer_depth > 0 is not supported")
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.heads = heads
        self.dim_head = dim_head
        tkw = dict(
            dim=dim, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout, norm_output=False
        )
        time_rotary = RotaryEmbedding(dim=dim_head)
        freq_rotary = RotaryEmbedding(dim=dim_head)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    Transformer(depth=time_transformer_depth, rotary_embed=time_rotary, **tkw),
                    Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary, **tkw),
                ]
            )
            for _ in range(depth)
        )
        self.final_norm = RMSNorm(dim)
        self.stft_kwargs = dict(
            n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length, normalized=stft_normalized
        )
        self.stft_window_fn = partial(torch.hann_window, stft_win_length)
        n_freqs = stft_n_fft // 2 + 1
        if sum(freqs_per_bands) != n_freqs:
            raise ValueError(f"freqs_per_bands must sum to {n_freqs} (n_fft={stft_n_fft}), got {sum(freqs_per_bands)}")
        dims = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim=dim, dim_inputs=dims)
        self.mask_estimators = nn.ModuleList(
            MaskEstimator(
                dim=dim, dim_inputs=dims, depth=mask_estimator_depth, mlp_expansion_factor=mlp_expansion_factor
            )
            for _ in range(num_stems)
        )

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)
        b, s, t = raw_audio.shape
        if s != self.audio_channels:
            raise ValueError(f"expected {self.audio_channels} audio channels, got {s}")
        window = self.stft_window_fn(device=raw_audio.device)
        spec = torch.view_as_real(
            torch.stft(raw_audio.reshape(b * s, t), **self.stft_kwargs, window=window, return_complex=True)
        )  # (b s, f, frames, c)
        _, nf, nt, _ = spec.shape
        # "b s f t c -> b (f s) t c", then "b f t c -> b t (f c)"
        spec = spec.view(b, s, nf, nt, 2).permute(0, 2, 1, 3, 4).reshape(b, nf * s, nt, 2)
        x = spec.permute(0, 2, 1, 3).reshape(b, nt, nf * s * 2)
        x = self.band_split(x)  # (b, t, bands, d)
        for time_tr, freq_tr in self.layers:
            nb, d = x.shape[2], x.shape[3]
            x = x.transpose(1, 2).reshape(b * nb, nt, d)  # "b t f d -> (b f) t d"
            x = time_tr(x)
            x = x.view(b, nb, nt, d).transpose(1, 2).reshape(b * nt, nb, d)  # "-> (b t) f d"
            x = freq_tr(x)
            x = x.view(b, nt, nb, d)
        x = self.final_norm(x)
        mask = torch.stack([est(x) for est in self.mask_estimators], dim=1)  # (b, n, t, f*c)
        n = mask.shape[1]
        mask = mask.view(b, n, nt, nf * s, 2).permute(0, 1, 3, 2, 4)  # "b n t (f c) -> b n f t c"
        out = torch.view_as_complex(spec.unsqueeze(1).contiguous()) * torch.view_as_complex(mask.contiguous())
        # "b n (f s) t -> (b n s) f t"
        out = out.view(b, n, nf, s, nt).permute(0, 1, 3, 2, 4).reshape(b * n * s, nf, nt)
        audio = torch.istft(out, **self.stft_kwargs, window=window, return_complex=False)
        audio = audio.view(b, n, s, -1)
        return audio[:, 0] if n == 1 else audio


def build_model(model_cfg: dict[str, Any]) -> BSRoformer:
    """Instantiate a BSRoformer from the ``model:`` section of a ZFTurbo/audio-separator style config."""
    cfg = dict(model_cfg)
    dim = cfg.pop("dim")
    cfg["freqs_per_bands"] = tuple(cfg.get("freqs_per_bands", DEFAULT_FREQS_PER_BANDS))
    return BSRoformer(dim, **cfg)


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    """Load a checkpoint (plain state dict, or wrapped in ``state_dict``/``model``) with strict key matching."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    return model.float().eval()
