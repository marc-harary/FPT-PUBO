# cc_bqp_swap_antisym_experiment.py
#
# Goal (your (i)-(vii)):
# (i)  generate cc-BQP with strictly positive entries
# (ii) brute-force global opt on hypersimplex (n choose k)
# (iii) show swap “antisymmetry” sign(δ_ij) = -sign(δ_ji) (incl. infeasible) need not hold
# (iv)  transform: make Q off-diagonal nonpositive, absorb diagonal into c, optionally add J-shift
# (v)   brute-force again
# (vi)  verify “anti-(-,-)” holds for ALL swap pairs after transform:
#          not( δ_ij<0 and δ_ji<0 )  for all i≠j   (equivalently δ_ij+δ_ji ≥ 0)
#       (this is what Q_ii+Q_jj-2Q_ij ≥ 0 guarantees, and after diag-absorb + Q_ij≤0 it’s automatic)
# (vii) compare optimal supports; compare objective values up to the known constant (only from J-shift)

import numpy as np
import itertools
from tqdm import tqdm


# -------------------------
# objective + brute force
# -------------------------
def eval_obj(Q, c, x):
    return x @ Q @ x + c @ x


def brute_force_cc_bqp(Q, c, k, tol=1e-12):
    n = Q.shape[0]
    best_val = -np.inf
    optima = []
    supports = []
    for comb in itertools.combinations(range(n), k):
        x = np.zeros(n, dtype=float)
        x[list(comb)] = 1.0
        v = eval_obj(Q, c, x)
        if v > best_val + tol:
            best_val = v
            optima = [x]
            supports = [frozenset(comb)]
        elif abs(v - best_val) <= tol:
            optima.append(x)
            supports.append(frozenset(comb))
    return best_val, optima, supports


# -------------------------
# “same-state” swap gains
# -------------------------
def delta_swap(Q, c, x, i, j):
    # δ_{ij}(x) := f(x - e_i + e_j) - f(x)
    # computed on R^n (so “infeasible” swaps included automatically)
    x_new = x.copy()
    x_new[i] -= 1.0
    x_new[j] += 1.0
    return eval_obj(Q, c, x_new) - eval_obj(Q, c, x)


def sign3(v, tol):
    if v > tol:
        return +1
    if v < -tol:
        return -1
    return 0


def check_full_antisymmetry(Q, c, x, tol=1e-9):
    # full antisymmetry (usually false):
    #   sign(δ_ij(x)) == -sign(δ_ji(x)) for all i≠j
    n = Q.shape[0]
    bad = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sij = sign3(delta_swap(Q, c, x, i, j), tol)
            sji = sign3(delta_swap(Q, c, x, j, i), tol)
            if sij != -sji:
                bad.append((i, j, sij, sji))
                if len(bad) >= 20:
                    return False, bad
    return True, bad


def check_anti_minusminus(Q, c, x, tol=1e-12):
    # anti-(-,-) (what your “triangle rule” blocks):
    #   NOT( δ_ij(x) < 0 AND δ_ji(x) < 0 ) for all i≠j
    # equivalently δ_ij(x)+δ_ji(x) >= 0 when evaluated exactly.
    n = Q.shape[0]
    bad = []
    for i in range(n):
        for j in range(i + 1, n):
            dij = delta_swap(Q, c, x, i, j)
            dji = delta_swap(Q, c, x, j, i)
            if (dij < -tol) and (dji < -tol):
                bad.append((i, j, dij, dji))
                if len(bad) >= 20:
                    return False, bad
    return True, bad


# -------------------------
# transform: force Q_ij <= 0 and absorb diag(Q) into c
# -------------------------
def make_offdiag_nonpos_and_absorb_diag(Q, c):
    """
    Produce (Q2, c2) s.t.
      - Q2 has zero diagonal (absorbed into c2),
      - Q2 off-diagonals are <= 0 (elementwise),
      - the objective is IDENTICAL on {0,1}^n up to an optional constant from a J-shift.

    Construction:
      1) absorb diag: c <- c + diag(Q), set diag(Q)=0
      2) shift off-diagonals down by their maximum so all off-diagonals become <= 0:
           let M = max_{i≠j} Q_ij  (after diag removed)
           set Q <- Q - M*(J - I)
         This adds a cardinality-only constant on the hypersimplex:
           x^T (J-I) x = k^2 - k   for 1^T x = k
         so the argmax over {x:1^T x=k} is invariant, and values shift by -M*(k^2-k).

    Returns (Q2, c2, M) where M is the offdiag shift used.
    """
    Q1 = Q.copy()
    c1 = c.copy()

    # (iv.a) absorb diagonal exactly (since x_i^2 = x_i on {0,1})
    diag = np.diag(Q1).copy()
    c1 = c1 + diag
    np.fill_diagonal(Q1, 0.0)

    # (iv.b) force off-diagonal <= 0 by shifting down by max offdiag
    n = Q1.shape[0]
    offdiag_mask = ~np.eye(n, dtype=bool)
    M = np.max(Q1[offdiag_mask])
    # after subtracting, new offdiag = old - M <= 0
    JminusI = np.ones((n, n), dtype=float) - np.eye(n, dtype=float)
    Q2 = Q1 - M * JminusI

    # c unchanged in this step; only constant shift on fixed k
    c2 = c1
    return Q2, c2, M


# -------------------------
# random positive instance
# -------------------------
def random_positive_instance(n, scale=5.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    A = rng.uniform(0.0, scale, size=(n, n))
    Q = 0.5 * (A + A.T)
    c = rng.uniform(0.0, scale, size=n)
    return Q, c


def main():
    master_rng = np.random.default_rng(0)

    n = 12
    k = 6
    n_trials = 100

    tol_sign = 1e-9
    tol_val = 1e-8

    seeds = [master_rng.integers(1_000_000) for _ in range(n_trials)]

    stats = dict(
        full_antisym_pass_before=0,
        anti_mm_pass_before=0,
        full_antisym_pass_after=0,
        anti_mm_pass_after=0,
        argmax_match=0,
    )

    first_cex = None

    for t in tqdm(range(n_trials)):
        rng = np.random.default_rng(seeds[t])

        # (i) positive instance
        Q, c = random_positive_instance(n, rng=rng)

        # (ii) solve original
        best0, opt0, supp0 = brute_force_cc_bqp(Q, c, k)
        supp0_set = set(supp0)
        xstar0 = opt0[0]  # representative optimum

        # (iii) antisymmetry checks at x*
        ok_full0, bad_full0 = check_full_antisymmetry(Q, c, xstar0, tol=tol_sign)
        ok_mm0, bad_mm0 = check_anti_minusminus(Q, c, xstar0, tol=tol_sign)
        stats["full_antisym_pass_before"] += int(ok_full0)
        stats["anti_mm_pass_before"] += int(ok_mm0)

        # (iv) transform: offdiag <= 0, absorb diag, track constant shift on fixed k
        Q2, c2, M = make_offdiag_nonpos_and_absorb_diag(Q, c)
        # constant shift on hypersimplex:
        # x^T(-M(J-I))x = -M*(k^2-k)
        const_shift = -M * (k * k - k)

        # (v) solve transformed
        best2, opt2, supp2 = brute_force_cc_bqp(Q2, c2, k)
        supp2_set = set(supp2)
        xstar2 = opt2[0]

        # (vi) checks after transform
        ok_full2, bad_full2 = check_full_antisymmetry(Q2, c2, xstar2, tol=tol_sign)
        ok_mm2, bad_mm2 = check_anti_minusminus(Q2, c2, xstar2, tol=tol_sign)
        stats["full_antisym_pass_after"] += int(ok_full2)
        stats["anti_mm_pass_after"] += int(ok_mm2)

        # (vii) verify argmax invariance and value shift
        # Values should satisfy: best2 == best0 + const_shift (up to tol),
        # because diag-absorb is exact (no constant), and offdiag shift is constant on k.
        if abs(best2 - (best0 + const_shift)) > tol_val:
            print("\nVALUE MISMATCH")
            print("seed =", seeds[t])
            print("best0 =", best0)
            print("best2 =", best2)
            print("expected =", best0 + const_shift)
            return

        if supp0_set == supp2_set:
            stats["argmax_match"] += 1
        else:
            # ties can reshuffle: still should match as SETS of optimal supports if transform is truly constant on k
            # so mismatch here is a real red flag.
            print("\nARGMAX SUPPORT SET MISMATCH")
            print("seed =", seeds[t])
            print("len(supp0) =", len(supp0_set), "len(supp2) =", len(supp2_set))
            return

        # record first counterexample to full antisym (before), and show after anti-(-,-)
        if first_cex is None and (not ok_full0):
            first_cex = dict(
                seed=int(seeds[t]),
                xstar=xstar0.astype(int).tolist(),
                bad_full0=bad_full0[:10],
                ok_mm2=bool(ok_mm2),
                bad_mm2=bad_mm2[:10],
                M=float(M),
            )

    print("\n" + "=" * 72)
    print(f"n={n} k={k} trials={n_trials}")
    print("Before transform (positive Q,c):")
    print(f"  full antisymmetry passes: {stats['full_antisym_pass_before']}/{n_trials}")
    print(f"  anti-(-,-) passes:        {stats['anti_mm_pass_before']}/{n_trials}")
    print("After transform (diag absorbed, offdiag shifted so Q_ij<=0):")
    print(f"  full antisymmetry passes: {stats['full_antisym_pass_after']}/{n_trials}")
    print(f"  anti-(-,-) passes:        {stats['anti_mm_pass_after']}/{n_trials}")
    print("Argmax support-set equality (should be all trials):")
    print(f"  matches: {stats['argmax_match']}/{n_trials}")

    if first_cex is not None:
        print("\nFirst counterexample snapshot:")
        print("  seed:", first_cex["seed"])
        print("  x*:", first_cex["xstar"])
        print("  first few (i,j,sij,sji) violating full antisym (before):", first_cex["bad_full0"])
        print("  shift M used to force offdiag<=0:", first_cex["M"])
        print("  anti-(-,-) after transform:", first_cex["ok_mm2"])
        if not first_cex["ok_mm2"]:
            print("  first few (-,-) swap pairs after:", first_cex["bad_mm2"])


if __name__ == "__main__":
    main()
