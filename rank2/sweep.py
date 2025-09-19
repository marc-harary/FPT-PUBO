#!/usr/bin/env python3
"""
Rank-2 QUBO (2D predicate sweep via Shapely) — self-test against brute force

Problem: maximize  f(x) = (v1^T x)^2 + (v2^T x)^2 + d^T x,  x in {0,1}^n.
Method: use Rank2QUBO (adjacent module) to build a Shapely arrangement of
        affine predicates g_i^+(ξ)=a_i ξ1 + b_i ξ2 + c_i, enumerate bounded
        faces, sample one ξ per face, map sign pattern → support, evaluate f,
        and take the best. Compare to brute force.

Usage:
    python rank2_sweep_selftest.py --n 14 --niter 200 --seed 123 --bbox-scale 10.0

Requirements:
    - shapely (installed in the environment)
    - An adjacent file `rank2_qubo.py` exporting `Rank2QUBO`
      (i.e., `from rank2_qubo import Rank2QUBO` works from this script's folder).

9/19/25
"""

from __future__ import annotations
from typing import Tuple
import argparse
import numpy as np
from tqdm import tqdm

# Import the class from the adjacent file (same directory)
from rank2_qubo import Rank2QUBO


# ---------- Ground truth (small n) ----------
def brute_best_rank2(
    v1: np.ndarray, v2: np.ndarray, d: np.ndarray
) -> Tuple[float, np.ndarray, Tuple[float, float]]:
    """
    Exhaustive maximization of f(x) = (v1^T x)^2 + (v2^T x)^2 + d^T x over x ∈ {0,1}^n.
    Returns (best_value, best_x, (xi1, xi2)).
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    d = np.asarray(d, dtype=float)
    n = v1.size

    best_val = -np.inf
    best_x = np.zeros(n, dtype=int)
    best_U = (0.0, 0.0)

    for mask in range(1 << n):
        x = np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=int, count=n)
        xi1 = float(v1 @ x)
        xi2 = float(v2 @ x)
        val = (xi1 * xi1) + (xi2 * xi2) + float(d @ x)
        if val > best_val:
            best_val = val
            best_x = x
            best_U = (xi1, xi2)
    return best_val, best_x, best_U


# ---------- Trials ----------
def random_instance(
    n: int, rng: np.random.Generator, dist: str = "gauss"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dist == "gauss":
        v1 = rng.standard_normal(n)
        v2 = rng.standard_normal(n)
        # Optional: avoid near-collinearity for numerical hygiene
        if (
            abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            > 0.999
        ):
            v2 = rng.standard_normal(n)
        d = rng.normal(scale=0.2, size=n)
        return v1, v2, d
    if dist == "student3":
        v1 = rng.standard_t(df=3.0, size=n)
        v2 = rng.standard_t(df=3.0, size=n)
        d = rng.standard_t(df=3.0, size=n) * 0.2
        return v1, v2, d
    raise ValueError("unknown dist")


def sweep_rank2(
    v1: np.ndarray, v2: np.ndarray, d: np.ndarray, bbox_scale: float = 10.0
) -> Tuple[float, np.ndarray, Tuple[float, float]]:
    qubo = Rank2QUBO(v1, v2, d, bbox_scale=bbox_scale)
    return qubo.solve_by_faces()


def run_trials(n: int, niter: int, seed: int, bbox_scale: float = 10.0) -> int:
    """
    Run niter random trials; return index of first mismatch (sweep vs brute),
    or -1 if all trials match.
    """
    rng = np.random.default_rng(seed)
    for t in tqdm(range(niter)):
        v1, v2, d = random_instance(n, rng)
        s_b, x_b, _ = brute_best_rank2(v1, v2, d)
        s_s, x_s, _ = sweep_rank2(v1, v2, d, bbox_scale=bbox_scale)
        if not (np.isclose(s_b, s_s) and np.array_equal(x_b, x_s)):
            return t
    return -1


# ---------- Argparse main (mirrors rank-1 style) ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank-2 QUBO predicate face-sweep self-test (Shapely)"
    )
    ap.add_argument(
        "--n", type=int, default=14, help="dimension n (brute force is 2^n)"
    )
    ap.add_argument("--niter", type=int, default=200, help="number of random trials")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument(
        "--bbox-scale",
        type=float,
        default=10.0,
        help="window size multiplier for arrangement",
    )
    ap.add_argument(
        "--dist",
        type=str,
        default="gauss",
        choices=["gauss", "student3"],
        help="random instance distribution",
    )
    args = ap.parse_args()

    bad = run_trials(
        n=args.n, niter=args.niter, seed=args.seed, bbox_scale=args.bbox_scale
    )
    if bad >= 0:
        print(
            f"Violation found at trial index {bad} with n={args.n}, "
            f"niter={args.niter}, seed={args.seed}, bbox_scale={args.bbox_scale}"
        )
    else:
        print(
            f"All {args.niter} trials passed for n={args.n}, seed={args.seed}, bbox_scale={args.bbox_scale}"
        )


if __name__ == "__main__":
    main()
