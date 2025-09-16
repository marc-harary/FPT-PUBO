#!/usr/bin/env python3
"""
Rank-1 QUBO (1D predicate sweep) – exact via root enumeration

Problem: maximize  f(x) = lam * (v^T x)^2,  x in {0,1}^n.
Method: compute predicate roots r_i where add-gain g_i^+(ξ) changes sign; sort roots;
sample one ξ per interval; in each interval, set x_i = 1[g_i^+(ξ) > 0]; evaluate f(x);
return the best; compare to brute force.

9/16/25
"""

from __future__ import annotations
from typing import Tuple, List, Set
import argparse
import numpy as np
from tqdm import tqdm
from rank1.shared import brute_best_rank1


# ---------- Predicate machinery ----------
def add_gain(v: np.ndarray, lam: float, xi: float) -> np.ndarray:
    """g_i^+(ξ) = lam * (2 v_i ξ + v_i^2)"""
    return lam * (2.0 * v * xi + v**2)


def roots_rank1(v: np.ndarray, lam: float) -> np.ndarray:
    """
    Roots r_i solving g_i^+(ξ)=0 ⇒ 2 v_i ξ + v_i^2 = 0  (lam cancels if lam≠0)
    ⇒ r_i = -v_i/2  for v_i ≠ 0; NaN if v_i == 0 (predicate flat at 0).
    """
    v = np.asarray(v, dtype=float)
    r = np.full_like(v, np.nan, dtype=float)
    nz = (v != 0.0)
    r[nz] = -0.5 * v[nz]
    return r


def interval_representatives(roots: np.ndarray, pad: float = 1.0) -> List[float]:
    """
    Given roots r_1..r_m (with NaNs allowed), return one probe ξ per open interval:
      (-∞, r_(1)), (r_(1), r_(2)), …, (r_(m-1), r_(m)), (r_(m), +∞).
    Infinite intervals realized by extrapolating beyond min/max by a margin.
    """
    finite = np.sort(roots[np.isfinite(roots)])
    reps: List[float] = []
    if finite.size == 0:
        reps.append(0.0)
        return reps
    spread = float(finite[-1] - finite[0]) if finite.size > 1 else 1.0
    margin = pad * max(1.0, spread)
    reps.append(float(finite[0] - margin))  # far left
    for a, b in zip(finite[:-1], finite[1:]):
        reps.append(float(0.5 * (a + b)))
    reps.append(float(finite[-1] + margin))  # far right
    return reps


def support_from_probe(v: np.ndarray, lam: float, xi_probe: float) -> np.ndarray:
    """
    At probe ξ, set x_i = 1 if g_i^+(ξ) > 0 else 0. Ties (==0) → 0.
    """
    gplus = add_gain(v, lam, xi_probe)
    return (gplus > 0.0).astype(int)


def sweep_rank1(v: np.ndarray, lam: float = 1.0) -> Tuple[float, np.ndarray, float]:
    """
    Exact solver via predicate-root sweep (1D arrangement).
    Returns (best_score, best_x, xi_best).
    """
    v = np.asarray(v, dtype=float)
    r = roots_rank1(v, lam)
    reps = interval_representatives(r)

    best_score = -np.inf
    best_x = np.zeros_like(v, dtype=int)
    best_xi = 0.0

    seen: Set[tuple[int, ...]] = set()
    for xi in reps:
        x = support_from_probe(v, lam, xi)
        key = tuple(int(t) for t in x)
        if key in seen:
            continue
        seen.add(key)
        xi_val = float(v @ x)
        score = lam * (xi_val * xi_val)
        if score > best_score:
            best_score, best_x, best_xi = score, x, xi_val
    return best_score, best_x, best_xi


# ---------- Trials ----------
def random_instance(n: int, rng: np.random.Generator, dist: str = "gauss") -> np.ndarray:
    if dist == "gauss":
        return rng.standard_normal(n)
    if dist == "student3":
        return rng.standard_t(df=3.0, size=n)
    raise ValueError("unknown dist")


def run_trials(n: int, niter: int, lam: float, seed: int) -> int:
    """
    Run niter random trials; return index of first mismatch (sweep vs brute),
    or -1 if all trials match.
    """
    rng = np.random.default_rng(seed)
    for t in tqdm(range(niter)):
        v = random_instance(n, rng)
        s_b, x_b, xi_b = brute_best_rank1(v, lam=lam)
        s_s, x_s, xi_s = sweep_rank1(v, lam=lam)
        if not (np.isclose(s_b, s_s) and np.array_equal(x_b, x_s)):
            # Optional: print debug details here if desired.
            return t
    return -1


# ---------- Argparse main (as requested) ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Rank-1 QUBO predicate self-test")
    ap.add_argument("--n", type=int, default=10, help="dimension n")
    ap.add_argument("--niter", type=int, default=10_000, help="number of random trials")
    ap.add_argument("--lam", type=float, default=1.0, help="lambda in objective")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    args = ap.parse_args()

    bad = run_trials(n=args.n, niter=args.niter, lam=args.lam, seed=args.seed)
    # CI/CLI-friendly exit condition; no tqdm or prints per-iteration
    if bad >= 0:
        # Nonzero exit code signals failure in CI if you choose to raise SystemExit(1)
        print(
            f"Violation found at trial index {bad} with n={args.n}, niter={args.niter}, lam={args.lam}, seed={args.seed}"
        )
    else:
        print(
            f"All {args.niter} trials passed for n={args.n}, lam={args.lam}, seed={args.seed}"
        )


if __name__ == "__main__":
    main()
