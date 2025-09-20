#!/usr/bin/env python3
"""
Predicate sanity tests with tqdm:
- Generate random QUBO instances (rank-k).
- Compute brute-force maximizer x* for f(x)=||Vx||^2 + d^T x.
- Verify 1-optimality predicates at U*=V x*:
    Add (x_i=0):  g_i^+(U*) =  2 v_i^T U* + ||v_i||^2 + d_i <= tol
    Drop(x_i=1):  g_i^-(U*) = -2 v_i^T U* + ||v_i||^2 - d_i <= tol
  where v_i is column i of V.

Run:
    python test_predicates.py
or with pytest:
    pytest -q test_predicates.py
"""

from __future__ import annotations
import unittest
import numpy as np
from tqdm.auto import tqdm


# ---------- helpers ----------

def random_instance(k: int, n: int, rng: np.random.Generator, dist: str = "gauss"):
    if dist == "gauss":
        V = rng.standard_normal((k, n))
        d = rng.normal(scale=0.2, size=n)
        return V, d
    if dist == "student3":
        V = rng.standard_t(df=3.0, size=(k, n))
        d = rng.standard_t(df=3.0, size=n) * 0.2
        return V, d
    raise ValueError("unknown dist")


def brute_best_rankk(V: np.ndarray, d: np.ndarray):
    """
    Exhaustive maximization of f(x)=||Vx||^2 + d^T x over x∈{0,1}^n.
    Returns (best_val, x*, U*=Vx*).
    """
    V = np.asarray(V, dtype=float)
    d = np.asarray(d, dtype=float)
    k, n = V.shape
    best_val = -np.inf
    best_x = np.zeros(n, dtype=int)
    best_U = np.zeros(k, dtype=float)
    for mask in tqdm(range(1 << n), desc="brute masks", leave=False):
        x = np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=int, count=n)
        U = V @ x
        val = float(U @ U + d @ x)
        if val > best_val:
            best_val, best_x, best_U = val, x, U
    return best_val, best_x, best_U


def predicate_mats(V: np.ndarray, d: np.ndarray):
    """
    Build predicate matrices for:
      g^+_i(U) = a_i^T U + c_add[i],   c_add[i]  = ||v_i||^2 + d_i
      g^-_i(U) = -a_i^T U + c_drop[i], c_drop[i] = ||v_i||^2 - d_i
    where a_i = 2 v_i (v_i is column i of V).
    """
    V = np.asarray(V, dtype=float)
    d = np.asarray(d, dtype=float)
    A = 2.0 * V.T                                 # shape (n,k)
    vv = np.einsum("ij,ij->j", V, V)              # ||v_i||^2 over columns
    c_add = vv + d
    c_drop = vv - d
    return A, c_add, c_drop


def scaled_tol(V: np.ndarray, d: np.ndarray, U: np.ndarray, base: float = 1e-10) -> float:
    """
    Scale tolerance to instance magnitude to avoid spurious FP flags.
    """
    Vn = float(np.linalg.norm(V, ord=2))
    Un = float(np.linalg.norm(U))
    dn = float(np.linalg.norm(d, ord=np.inf))
    return base * max(1.0, Vn * Un + dn)


def check_1_optimality(V: np.ndarray, d: np.ndarray, x: np.ndarray, U: np.ndarray, tol: float | None = None):
    """
    Assert 1-optimality at (x,U): no improving single-bit flip.
      For i with x_i=0: g_i^+(U) <= tol.
      For i with x_i=1: g_i^-(U) <= tol.
    Returns (max_add, max_drop) for diagnostics.
    """
    A, c_add, c_drop = predicate_mats(V, d)
    g_add  = A @ U + c_add
    g_drop = -A @ U + c_drop

    if tol is None:
        tol = scaled_tol(V, d, U, base=1e-10)

    x = x.astype(int, copy=False)
    add_viol_idx  = np.where((x == 0) & (g_add  > tol))[0]
    drop_viol_idx = np.where((x == 1) & (g_drop > tol))[0]

    if add_viol_idx.size or drop_viol_idx.size:
        msgs = []
        if add_viol_idx.size:
            msgs.append(f"add violations @ {add_viol_idx.tolist()} with max g^+={g_add[add_viol_idx].max():.3e}")
        if drop_viol_idx.size:
            msgs.append(f"drop violations @ {drop_viol_idx.tolist()} with max g^-={g_drop[drop_viol_idx].max():.3e}")
        raise AssertionError(" ; ".join(msgs))

    return float(g_add.max(initial=-np.inf)), float(g_drop.max(initial=-np.inf))


# ---------- tests ----------

class TestPredicates(unittest.TestCase):
    def test_rank1_small(self):
        rng = np.random.default_rng(123)
        k, n = 1, 15
        for _ in tqdm(range(25), desc="rank1 trials"):
            V, d = random_instance(k, n, rng)
            _, x, U = brute_best_rankk(V, d)
            check_1_optimality(V, d, x, U)

    def test_rank2_small(self):
        rng = np.random.default_rng(456)
        k, n = 4, 15
        for _ in tqdm(range(25), desc="rank2 trials"):
            V, d = random_instance(k, n, rng)
            _, x, U = brute_best_rankk(V, d)
            check_1_optimality(V, d, x, U)

    def test_rank2_small(self):
        rng = np.random.default_rng(456)
        k, n = 2, 16
        for _ in tqdm(range(25), desc="rank2 trials"):
            V, d = random_instance(k, n, rng)
            _, x, U = brute_best_rankk(V, d)
            check_1_optimality(V, d, x, U)

    def test_rankk_tiny(self):
        rng = np.random.default_rng(789)
        k, n = 3, 12  # brute force still OK
        for _ in tqdm(range(10), desc="rank3 trials"):
            V, d = random_instance(k, n, rng)
            _, x, U = brute_best_rankk(V, d)
            check_1_optimality(V, d, x, U)

    def test_student_t(self):
        rng = np.random.default_rng(7)
        k, n = 2, 14
        for _ in tqdm(range(10), desc="student-t trials"):
            V, d = random_instance(k, n, rng, dist="student3")
            _, x, U = brute_best_rankk(V, d)
            check_1_optimality(V, d, x, U)


if __name__ == "__main__":
    unittest.main()
