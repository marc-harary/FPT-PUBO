#!/usr/bin/env python3
"""
Rank-1 QUBO predicate check
9/16/25

Generates random symmetric rank-1 QUBOs, solves them exactly by brute force,
and verifies 1-optimality via affine predicates (add/drop) evaluated at ξ* = v^T x*.

Objective:
    maximize f(x) = λ (v^T x)^2,  x ∈ {0,1}^n

For rank-1, single-move gains are affine in ξ:
    g_i^+(ξ) = λ (2 v_i ξ + v_i^2)          # add predicate (x_i: 0→1)
    g_i^-(ξ) = λ (-2 v_i ξ + v_i^2)         # drop predicate (x_i: 1→0)
1-optimality (maximization):
    ∀i with x_i=0: g_i^+(ξ*) ≤ 0;   ∀j with x_j=1: g_j^-(ξ*) ≤ 0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import argparse
import numpy as np
from tqdm import tqdm
from rank1.shared import brute_best_rank1


@dataclass(frozen=True)
class PredicateReport:
    xi_star: float
    add_gain: np.ndarray
    drop_gain: np.ndarray
    add_ok_mask: np.ndarray  # valid where x==0
    drop_ok_mask: np.ndarray  # valid where x==1
    violating_add_idx: np.ndarray
    violating_drop_idx: np.ndarray
    all_ok: bool
    roots: np.ndarray  # ξ where g_i^+(ξ)=0 if v_i≠0; else NaN


def check_rank1_predicates(
    v: np.ndarray, lam: float, x_star: np.ndarray, xi_star: float
) -> PredicateReport:
    """
    Evaluate affine add/drop predicates at ξ* for candidate maximizer x*.
    See module docstring for definitions.
    """
    v = np.asarray(v, dtype=float)
    x = np.asarray(x_star, dtype=int)
    assert v.shape == x.shape and v.ndim == 1
    n = v.size
    xi = float(xi_star)

    add_gain = lam * (2.0 * v * xi + v**2)
    drop_gain = lam * (-2.0 * v * xi + v**2)

    in_set = x == 1
    out_set = ~in_set

    add_ok_mask = np.ones(n, dtype=bool)
    drop_ok_mask = np.ones(n, dtype=bool)

    add_ok_mask[out_set] = add_gain[out_set] <= 0.0
    drop_ok_mask[in_set] = drop_gain[in_set] <= 0.0

    violating_add_idx = np.nonzero(out_set & ~add_ok_mask)[0]
    violating_drop_idx = np.nonzero(in_set & ~drop_ok_mask)[0]

    roots = np.full(n, np.nan)
    nz = v != 0.0
    roots[nz] = -0.5 * v[nz]  # g_i^+(ξ)=0 ⇒ ξ= -v_i/2

    all_ok = (violating_add_idx.size == 0) and (violating_drop_idx.size == 0)

    return PredicateReport(
        xi_star=xi,
        add_gain=add_gain,
        drop_gain=drop_gain,
        add_ok_mask=add_ok_mask,
        drop_ok_mask=drop_ok_mask,
        violating_add_idx=violating_add_idx,
        violating_drop_idx=violating_drop_idx,
        all_ok=all_ok,
        roots=roots,
    )


def run_trials(n: int, niter: int, lam: float, seed: int) -> int:
    """
    Run niter random trials; return the index of the first violating seed or -1 if all pass.
    Uses a dedicated Generator to avoid global RNG side effects.
    """
    rng = np.random.default_rng(seed)
    for t in tqdm(range(niter)):
        v = rng.standard_normal(n)
        _, x_star, xi_star = brute_best_rank1(v, lam=lam)
        rep = check_rank1_predicates(v, lam, x_star, xi_star)
        if not rep.all_ok:
            return t
    return -1


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
