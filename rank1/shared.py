# fpt_pubo/shared.py
from __future__ import annotations
from typing import Tuple
import numpy as np

__all__ = ["brute_best_rank1"]

def brute_best_rank1(v: np.ndarray, lam: float = 1.0) -> Tuple[float, np.ndarray, float]:
    """
    Brute-force maximization of f(x) = lam * (v^T x)^2 over x ∈ {0,1}^n.

    Parameters
    ----------
    v : (n,) array_like
        Rank-1 factor vector.
    lam : float, default 1.0
        Scalar coefficient (can be positive or negative).

    Returns
    -------
    best_score : float
        Maximum objective value.
    best_x : (n,) np.ndarray[int]
        Maximizer with entries in {0,1}.
    best_xi : float
        ξ* = v^T best_x.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    best_score = -np.inf
    best_x = np.zeros(n, dtype=int)
    best_xi = 0.0
    for mask in range(1 << n):
        x = np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=int, count=n)
        xi = float(v @ x)
        score = lam * (xi * xi)
        if score > best_score:
            best_score, best_x, best_xi = score, x, xi
    return best_score, best_x, best_xi
