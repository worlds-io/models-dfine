"""Correctness tests for the GPU auction LAP implementation in src/zoo/dfine/matcher.py.

Compares the auction output against scipy.optimize.linear_sum_assignment on a variety of
cost-matrix shapes and value distributions, asserting matched-cost-sum parity (ties allowed,
so we check total cost, not index equality).

Skipped when CUDA is unavailable — the auction kernel is GPU-only.

Run:
    pytest tests/test_gpu_auction_lap.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zoo.dfine.matcher import _auction_match_from_flat_cost  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU auction requires CUDA")


def _scipy_cost(C_np: np.ndarray) -> float:
    row, col = linear_sum_assignment(np.nan_to_num(C_np, nan=1.0))
    return float(C_np[row, col].sum())


def _single_case(Q: int, T: int, *, seed: int, scale: float = 1.0, offset: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn((Q, T), generator=g) * scale + offset


@pytest.mark.parametrize("Q,T", [(50, 0), (50, 1), (50, 5), (50, 10), (50, 25), (50, 50)])
def test_random_shapes(Q: int, T: int) -> None:
    """Random costs with varied Q/T combos — the common DETR regime."""
    for seed in range(30):
        C = _single_case(Q, T, seed=seed)
        # Pad to batched form (B=1)
        sizes = [T]
        C_flat = C.cuda().view(Q, T)  # already matches expected [Q, T] block layout after view
        # Build [bs*Q, total_T] as forward() produces. With bs=1, total_T=T, this is [Q, T].
        indices = _auction_match_from_flat_cost(C_flat, num_queries=Q, sizes=sizes)
        assert len(indices) == 1
        rows, cols = indices[0]
        assert rows.shape == cols.shape == (T,)
        gpu_cost = float(C[rows.cpu(), cols.cpu()].sum())
        cpu_cost = _scipy_cost(C.numpy())
        assert abs(gpu_cost - cpu_cost) < 1e-3, (
            f"Q={Q} T={T} seed={seed}: gpu={gpu_cost:.4f} cpu={cpu_cost:.4f}"
        )


def test_pathological_all_equal() -> None:
    """All-equal costs — many ties, should still reach a valid assignment."""
    for T in (5, 10, 25):
        C = torch.ones(50, T)
        C_flat = C.cuda()
        indices = _auction_match_from_flat_cost(C_flat, num_queries=50, sizes=[T])
        rows, cols = indices[0]
        assert rows.shape == (T,) and cols.shape == (T,)
        # All assignments are equivalent; cost sum must be exactly T
        assert float(C[rows.cpu(), cols.cpu()].sum()) == float(T)
        # Each query and each target used at most once
        assert len(set(rows.tolist())) == T
        assert len(set(cols.tolist())) == T


def test_pathological_wide_range() -> None:
    """Costs spanning many orders of magnitude."""
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        C = torch.exp(torch.randn(50, 25, generator=g) * 5)  # values in [~1e-8, ~1e8]
        C_flat = C.cuda()
        indices = _auction_match_from_flat_cost(C_flat, num_queries=50, sizes=[25])
        rows, cols = indices[0]
        gpu_cost = float(C[rows.cpu(), cols.cpu()].sum())
        cpu_cost = _scipy_cost(C.numpy())
        # Tolerance scales with cost magnitude
        tol = max(1e-3, 1e-4 * cpu_cost)
        assert abs(gpu_cost - cpu_cost) < tol, f"seed={seed} gpu={gpu_cost} cpu={cpu_cost}"


def test_pathological_duplicated_rows() -> None:
    """Many identical queries — creates tie explosions."""
    for T in (5, 25):
        C = torch.randn(50, T).float()
        C[0] = C[1] = C[2] = C[3]  # first 4 queries identical
        C_flat = C.cuda()
        indices = _auction_match_from_flat_cost(C_flat, num_queries=50, sizes=[T])
        rows, cols = indices[0]
        gpu_cost = float(C[rows.cpu(), cols.cpu()].sum())
        cpu_cost = _scipy_cost(C.numpy())
        assert abs(gpu_cost - cpu_cost) < 1e-3


def test_batched_multiple_images() -> None:
    """Batched call with 16 problems of varied T matches per-image scipy."""
    bs = 16
    Q = 50
    sizes = [0, 1, 3, 5, 10, 15, 20, 25, 30, 2, 8, 12, 50, 7, 4, 40]
    assert len(sizes) == bs
    g = torch.Generator().manual_seed(42)
    per_image_costs = [torch.randn(Q, t, generator=g) if t > 0 else torch.zeros(Q, 0) for t in sizes]
    # Build [bs*Q, total_T] flat cost, concatenating along the target axis in image order
    total_T = sum(sizes)
    C_flat = torch.zeros(bs * Q, total_T)
    offset = 0
    for i, (t, C_i) in enumerate(zip(sizes, per_image_costs)):
        if t > 0:
            C_flat[i * Q : (i + 1) * Q, offset : offset + t] = C_i
        offset += t

    indices = _auction_match_from_flat_cost(C_flat.cuda(), num_queries=Q, sizes=sizes)
    assert len(indices) == bs
    for i, (t, C_i) in enumerate(zip(sizes, per_image_costs)):
        rows, cols = indices[i]
        if t == 0:
            assert rows.numel() == 0 and cols.numel() == 0
            continue
        gpu_cost = float(C_i[rows.cpu(), cols.cpu()].sum())
        cpu_cost = _scipy_cost(C_i.numpy())
        assert abs(gpu_cost - cpu_cost) < 1e-3, (
            f"image {i} t={t}: gpu={gpu_cost:.4f} cpu={cpu_cost:.4f}"
        )


def test_auction_returns_valid_assignment() -> None:
    """Each query / target appears at most once in the returned indices."""
    C = torch.randn(50, 25)
    C_flat = C.cuda()
    indices = _auction_match_from_flat_cost(C_flat, num_queries=50, sizes=[25])
    rows, cols = indices[0]
    # One-to-one: unique pred indices, unique target indices, matching count
    assert rows.numel() == cols.numel() == 25
    assert len(set(rows.tolist())) == 25
    assert len(set(cols.tolist())) == 25


if __name__ == "__main__":  # pragma: no cover
    # Allow running directly for quick debugging without pytest infrastructure
    sys.exit(pytest.main([__file__, "-q", "-x"]))
