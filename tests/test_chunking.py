"""Chunk schedule and overlap-add: coverage, exact reconstruction, no boundary clicks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from turbo_roformer.chunking import chunk_starts, hamming, overlap_add, step_for


def test_schedule_matches_audio_separator_example():
    # 4:12 song at 44.1 kHz with the resurrection chunk (749259) and overlap 2
    n, chunk = 11130532, 749259
    starts = chunk_starts(n, chunk, step_for(chunk, 2))
    assert step_for(chunk, 2) == 374629
    assert len(starts) == 29
    assert starts[:3] == [0, 374629, 749258]
    assert starts[-1] == n - chunk


@pytest.mark.parametrize("n", [1, 100, 999, 1000, 1001, 5000, 12345])
@pytest.mark.parametrize("overlap", [1.0, 1.5, 2.0, 3.0])
def test_schedule_covers_signal(n, overlap):
    chunk = 1000
    C = min(chunk, n)
    starts = chunk_starts(n, C, step_for(C, overlap))
    covered = np.zeros(n, bool)
    for s in starts:
        assert 0 <= s <= n - C
        covered[s : s + C] = True
    assert covered.all()
    assert starts == sorted(set(starts))


def test_hamming_matches_numpy():
    for n in (2, 7, 1000):
        np.testing.assert_allclose(hamming(n), np.hamming(n), rtol=0, atol=1e-6)
    assert hamming(1000).min() > 0.07


@pytest.mark.parametrize("n", [700, 1000, 4321, 10007])
@pytest.mark.parametrize("overlap", [1.0, 1.5, 2.0, 3.0])
def test_identity_model_reconstructs_input(n, overlap):
    x = torch.randn(2, n)
    y = overlap_add(x, 1000, overlap, lambda c: c)
    torch.testing.assert_close(y, x, rtol=0, atol=1e-5)


@pytest.mark.parametrize("overlap", [1.5, 2.0, 3.0])
def test_no_click_with_chunk_dependent_model(overlap):
    """Chunks that disagree (here: +-20 % gain) must be crossfaded, not switched.

    With a DC input the output is the blending envelope itself. A hard switch would step by 0.4; the windowed
    overlap-add only steps where a Hamming window starts/ends at 0.08 of its peak (same as audio-separator),
    which is a small fraction of the chunk disagreement.
    """
    n, chunk = 100_000, 16000
    x = torch.ones(2, n)
    gains = iter([0.8, 1.2] * 100)
    y = overlap_add(x, chunk, overlap, lambda c: c * next(gains)).numpy()
    assert y.min() >= 0.8 - 1e-6 and y.max() <= 1.2 + 1e-6
    max_step = np.abs(np.diff(y, axis=1)).max()
    assert max_step < 0.25 * 0.4
    hard = overlap_add(x, chunk, 1.0, lambda c: c * next(gains)).numpy()  # no overlap -> visible switch
    assert np.abs(np.diff(hard, axis=1)).max() > 0.39


def test_progress_callback_counts():
    calls = []
    overlap_add(torch.randn(2, 5000), 1000, 2.0, lambda c: c, progress=lambda k, n: calls.append((k, n)))
    n = len(chunk_starts(5000, 1000, 500))
    assert calls == [(k, n) for k in range(1, n + 1)]
