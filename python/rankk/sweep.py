#!/usr/bin/env python3
"""
Fixed-k hyperplane-arrangement sampler (robust) for QUBO sanity checks
- Key fixes: tau=0 (ties→0 only), column scaling of A, adaptive radii via ||A_S^{-T}||,
  pseudoinverse point on ridges, and a couple of jittered directions per vertex.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple
import argparse
import itertools as it
import math
import numpy as np
from numpy.linalg import lstsq, svd, solve, pinv, norm
from tqdm.auto import tqdm


# ---------- Objective & predicates ----------

def objective(V: np.ndarray, d: np.ndarray, x: np.ndarray) -> Tuple[float, np.ndarray]:
    x = x.astype(float, copy=False)
    U = V @ x
    return float(U @ U + d @ x), U

def predicate_values(V: np.ndarray, d: np.ndarray, U: np.ndarray) -> np.ndarray:
    return 2.0 * (V.T @ U) + np.einsum("ij,ij->j", V, V) + d

def support_from_U(V: np.ndarray, d: np.ndarray, U: np.ndarray) -> np.ndarray:
    # tau = 0: ties→0 only at exact zero
    g = predicate_values(V, d, U)
    return (g > 0.0).astype(int)


# ---------- Sampler ----------

@dataclass
class ArrangementSampler:
    V: np.ndarray      # (k, n)
    d: np.ndarray      # (n,)
    eps: float = 1e-6
    cond_tol: float = 1e12
    radii: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    cap_vertices: Optional[int] = None
    cap_ridges: Optional[int] = None
    seed: Optional[int] = 123
    jitter_count: int = 2      # random directions per vertex

    def __post_init__(self):
        V = np.asarray(self.V, dtype=float); d = np.asarray(self.d, dtype=float)
        if V.ndim != 2 or d.ndim != 1 or d.shape[0] != V.shape[1]:
            raise ValueError("shapes: V=(k,n), d=(n,)")
        self.V, self.d = V, d
        self.k, self.n = V.shape

        # Build A, c_add
        A = 2.0 * V.T                                  # (n×k)
        c_add = np.einsum("ij,ij->j", V, V) + d        # (n,)

        # Column scaling: scale columns of A to unit 2-norm for solves;
        # keep transform to map between scaled and unscaled U.
        col_norms = norm(A, axis=0)
        col_norms[col_norms == 0] = 1.0
        self.S = 1.0 / col_norms                       # scale factors per column
        self.Sinv = col_norms

        self.As = A * self.S[None, :]                  # scaled A (n×k)
        self.c_add = c_add
        self.rng = np.random.default_rng(self.seed)

    def _local_length(self, A_Ss: np.ndarray) -> float:
        # Use ||A_S^{-T}||_2 ≈ 1/smin(A_S)
        s = svd(A_Ss, compute_uv=False)
        smin = float(np.min(s)) if s.size else 0.0
        return (1.0 / smin) if smin > 0 else np.inf

    def _unscale_U(self, Us_scaled: np.ndarray) -> np.ndarray:
        # U_unscaled = diag(Sinv) * Us_scaled
        return Us_scaled * self.Sinv

    # ----- Vertex probes -----

    def _vertex_probes(self) -> Iterable[np.ndarray]:
        n, k = self.n, self.k
        combos = it.combinations(range(n), k)
        if self.cap_vertices is not None:
            combos = it.islice(combos, self.cap_vertices)
        total = math.comb(n, k) if self.cap_vertices is None else None

        for S in tqdm(combos, total=total, desc=f"vertices (k={k})", leave=False):
            S = tuple(S)
            A_Ss = self.As[list(S), :]                 # scaled (k×k)
            c_S  = self.c_add[list(S)]                # (k,)

            # Rank & conditioning
            try:
                s = svd(A_Ss, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            smin, smax = float(np.min(s)), float(np.max(s))
            if not (smin > 0):
                continue
            cond = smax / smin
            if cond > self.cond_tol:
                continue

            # Vertex in scaled coords: A_Ss U*_s = -c_S
            try:
                U_star_s = solve(A_Ss, -c_S)
            except np.linalg.LinAlgError:
                continue

            # Local length scale
            ell = self._local_length(A_Ss)
            if not np.isfinite(ell):
                continue

            ATs = A_Ss.T
            # Cone directions: A_Ss^T δ_s = s * (eps*rmult*ell)
            for signs in it.product((-1.0, 1.0), repeat=k):
                svec = np.fromiter(signs, dtype=float, count=k)
                for rmult in self.radii:
                    rhs = svec * (self.eps * rmult * ell)
                    try:
                        delta_s = solve(ATs, rhs)
                    except np.linalg.LinAlgError:
                        continue
                    U_probe = self._unscale_U(U_star_s + delta_s)
                    yield U_probe

            # Jittered directions (to break symmetries / near-parallel issues)
            for _ in range(self.jitter_count):
                svec = self.rng.standard_normal(k)
                svec = svec / (norm(svec) + 1e-15)
                rmult = self.radii[1]  # ~eps*1*ell
                rhs = svec * (self.eps * rmult * ell)
                try:
                    delta_s = solve(ATs, rhs)
                except np.linalg.LinAlgError:
                    continue
                U_probe = self._unscale_U(U_star_s + delta_s)
                yield U_probe

    # ----- Ridge probes -----

    def _ridge_probes(self) -> Iterable[np.ndarray]:
        n, k = self.n, self.k
        if k == 1:
            return
        combos = it.combinations(range(n), k - 1)
        if self.cap_ridges is not None:
            combos = it.islice(combos, self.cap_ridges)
        total = math.comb(n, k - 1) if self.cap_ridges is None else None

        for S in tqdm(combos, total=total, desc=f"ridges (k-1={k-1})", leave=False):
            S = tuple(S)
            A_Rs = self.As[list(S), :]     # ((k-1)×k)
            c_R  = self.c_add[list(S)]     # ((k-1),)

            # Rank k-1?
            try:
                Uu, s, VT = svd(A_Rs, full_matrices=True)
            except np.linalg.LinAlgError:
                continue
            rank = int(np.sum(s > 0))
            if rank < k - 1:
                continue

            # Null direction (unit)
            n_Ss = VT[-1, :]
            n_Ss /= (norm(n_Ss) + 1e-15)

            # Particular point exactly on ridge in scaled coords: A_Rs U_Rs = -c_R (min-norm via pinv)
            U_Rs = pinv(A_Rs) @ (-c_R)

            # Adaptive radii
            ell = self._local_length(A_Rs)  # proxy using smin
            if not np.isfinite(ell):
                continue
            for rmult in self.radii:
                step = self.eps * rmult * ell
                U_probe_p = self._unscale_U(U_Rs + step * n_Ss)
                U_probe_m = self._unscale_U(U_Rs - step * n_Ss)
                yield U_probe_p
                yield U_probe_m

    # ----- Far-away probe -----

    def _far_probe(self) -> np.ndarray:
        normals = self.As.copy() * self.Sinv[None, :]  # unscale back to original metric
        norms = norm(normals, axis=1)
        if np.any(norms > 0):
            u_hat = (normals[norms > 0] / norms[norms > 0, None]).sum(axis=0)
        else:
            u_hat = np.zeros(self.k); u_hat[0] = 1.0
        u_hat /= (norm(u_hat) + 1e-15)
        R = 10.0 * (1.0 + float(np.max(np.abs(self.c_add))))
        return R * u_hat

    # ----- Main enumeration -----

    def enumerate_supports(self) -> List[np.ndarray]:
        seen: Set[Tuple[int, ...]] = set()
        supports: List[np.ndarray] = []

        for U in self._vertex_probes():
            x = support_from_U(self.V, self.d, U)
            key = tuple(int(t) for t in x)
            if key not in seen:
                seen.add(key); supports.append(x)

        for U in self._ridge_probes():
            x = support_from_U(self.V, self.d, U)
            key = tuple(int(t) for t in x)
            if key not in seen:
                seen.add(key); supports.append(x)

        U_far = self._far_probe()
        x_far = support_from_U(self.V, self.d, U_far)
        key_far = tuple(int(t) for t in x_far)
        if key_far not in seen:
            seen.add(key_far); supports.append(x_far)

        return supports

    def solve(self) -> Tuple[float, np.ndarray, np.ndarray]:
        best_val = -np.inf
        best_x = np.zeros(self.n, dtype=int)
        best_U = np.zeros(self.k, dtype=float)
        for x in tqdm(self.enumerate_supports(), desc="evaluate supports", leave=False):
            val, U = objective(self.V, self.d, x)
            if val > best_val:
                best_val, best_x, best_U = val, x, U
        return best_val, best_x, best_U


# ---------- Brute force & random ----------

def brute_best_rankk(V: np.ndarray, d: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    V = np.asarray(V, dtype=float); d = np.asarray(d, dtype=float)
    k, n = V.shape
    best_val = -np.inf; best_x = np.zeros(n, dtype=int); best_U = np.zeros(k)
    for mask in tqdm(range(1 << n), desc=f"brute masks (n={n})", leave=False):
        x = np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=int, count=n)
        val, U = objective(V, d, x)
        if val > best_val:
            best_val, best_x, best_U = val, x, U
    return best_val, best_x, best_U

def random_instance(k: int, n: int, rng: np.random.Generator, dist: str = "gauss") -> Tuple[np.ndarray, np.ndarray]:
    if dist == "gauss":
        V = rng.standard_normal((k, n)); d = rng.normal(scale=0.2, size=n); return V, d
    if dist == "student3":
        V = rng.standard_t(df=3.0, size=(k, n)); d = rng.standard_t(df=3.0, size=n) * 0.2; return V, d
    if dist == "uniform":
        V = rng.uniform(-1, 1, size=(k, n)); d = rng.uniform(-0.3, 0.3, size=n); return V, d
    raise ValueError("unknown dist")


# ---------- Trials / CLI ----------

def run_trials(k: int, n: int, niter: int, seed: int, eps: float, cond_tol: float,
               radii: Tuple[float, ...], cap_vertices: Optional[int], cap_ridges: Optional[int],
               dist: str, check_n: int, jitter_count: int) -> int:
    rng = np.random.default_rng(seed)
    for t in tqdm(range(niter), desc="trials"):
        V, d = random_instance(k, n, rng, dist=dist)
        sampler = ArrangementSampler(
            V, d, eps=eps, cond_tol=cond_tol, radii=radii,
            cap_vertices=cap_vertices, cap_ridges=cap_ridges,
            seed=seed + 17*t, jitter_count=jitter_count
        )
        s_s, x_s, _ = sampler.solve()
        if n <= check_n:
            s_b, x_b, _ = brute_best_rankk(V, d)
            if not (np.isclose(s_b, s_s) and np.array_equal(x_b, x_s)):
                return t
    return -1

def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed-k arrangement sampler for QUBO (robust NumPy)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--niter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--radii", type=str, default="0.1,1,10,100")
    ap.add_argument("--cond-tol", type=float, default=1e12)
    ap.add_argument("--cap-vertices", type=int, default=None)
    ap.add_argument("--cap-ridges", type=int, default=None)
    ap.add_argument("--dist", type=str, default="gauss", choices=["gauss","student3","uniform"])
    ap.add_argument("--check-n", type=int, default=15)
    ap.add_argument("--jitter", type=int, default=2, help="random directions per vertex")
    args = ap.parse_args()

    radii = tuple(float(s) for s in args.radii.split(",") if s.strip())

    bad = run_trials(
        k=args.k, n=args.n, niter=args.niter, seed=args.seed, eps=args.eps,
        cond_tol=args.cond_tol, radii=radii,
        cap_vertices=args.cap_vertices, cap_ridges=args.cap_ridges,
        dist=args.dist, check_n=args.check_n, jitter_count=args.jitter
    )
    if bad >= 0:
        print(f"Mismatch at trial {bad} with k={args.k}, n={args.n}, eps={args.eps}, radii={radii}, cond_tol={args.cond_tol}, seed={args.seed}")
    else:
        print(f"All {args.niter} trials finished; " +
              (f"all matched brute force (n≤{args.check_n})." if args.n <= args.check_n else
               "skipped brute-force (n too large)."))

if __name__ == "__main__":
    main()
