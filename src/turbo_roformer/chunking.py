"""Chunk schedule and windowed overlap-add (the same scheme python-audio-separator uses for Roformer models)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def chunk_starts(n: int, chunk: int, step: int) -> list[int]:
    """Regular steps; the tail chunk is aligned to the end of the signal (no zero-padded last chunk)."""
    starts: list[int] = []
    for off in range(0, n, step):
        if off + chunk >= n:
            tail = max(n - chunk, 0)
            if not starts or starts[-1] != tail:
                starts.append(tail)
            break
        starts.append(off)
    return starts


def hamming(n: int) -> np.ndarray:
    """Symmetric Hamming window (== scipy.signal.windows.hamming(n)); never zero, so every sample is covered."""
    if n == 1:
        return np.ones(1, np.float32)
    return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


def step_for(chunk: int, overlap: float) -> int:
    """step = floor(chunk / overlap): overlap 2 -> every sample is predicted ~2x."""
    return max(1, min(chunk, int(chunk / overlap)))


def overlap_add(
    x: torch.Tensor,
    chunk: int,
    overlap: float,
    run: Callable[[torch.Tensor], torch.Tensor],
    progress: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """x (2, n) -> (2, n). ``run`` maps a (1, 2, L) chunk to (1, 2, L') with L' <= L.

    Each chunk's prediction is weighted with a Hamming window and the sum is normalised by the summed weights,
    so any overlap >= 1 gives a seamless crossfade (no chunk-boundary clicks).
    """
    n = x.shape[1]
    C = min(chunk, n)
    starts = chunk_starts(n, C, step_for(C, overlap))
    w = torch.from_numpy(hamming(C))
    result = torch.zeros(2, n, dtype=torch.float32)
    weight = torch.zeros(n, dtype=torch.float32)
    for i, s in enumerate(starts):
        y = run(x[:, s : s + C].unsqueeze(0).contiguous())[0]
        L = min(C, y.shape[-1])
        result[:, s : s + L] += y[:, :L].float() * w[:L]
        weight[s : s + L] += w[:L]
        if progress is not None:
            progress(i + 1, len(starts))
    return result.div_(weight.clamp_(min=1e-10))
