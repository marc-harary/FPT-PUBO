# FPT-PUBO

Fixed-parameter tractable solver for Polynomial Unconstrained Binary Optimization via hyperplane arrangement enumeration. Shows that QUBO and ccBQP are XP in the rank of the coefficient matrix Q.

## julia/

### Core

- **`LowRankQUBO_IFS.jl`** — Main solver module. Provides:
  - `qubo_cc`: Cardinality-constrained BQP (ccBQP) via swap arrangements and SCC-based readout. For each swap direction `d = e_i - e_j`, the affine predicate `Δ_d(ξ) = v_d'ξ + c'd + d'Qd` determines whether swapping j out for i improves the objective. The exchange graph is built from predicate signs, and a k-closure is read out via condensation DAG.
  - `qubo_unconstrained`: Unconstrained QUBO via flip arrangements. Uses single-coordinate flip predicates `Δ_{e_i}(ξ) = 2(Λ ⊙ V[i,:])'ξ + c_i` with direct sign readout: `x_i = 1` iff `σ_{e_i}(ξ) > 0`. This works because the zero-diagonal assumption (`diag(Q) = 0`) ensures anti-symmetry of forward/backward flip predicates (`σ_{+d} = -σ_{-d}`), so a single predicate per coordinate suffices.

### Demos — Status

- **`demo.jl`** ✅ — Cardinality-constrained QUBO. 80 trials (n∈{6,8,10}, r=2, k∈{3,4,5}), all passing. Directly uses low-rank factors `V, Λ` since swap predicates involve `d'Qd = -2Q_{ij}` which does not require zero diagonal.
- **`demo_unconstrained.jl`** ✅ — Unconstrained QUBO. 200 trials (n∈{6,8,10,12}, r=2), all passing. **Requires the zero-diagonal trick**: given `Q = VΛV'`, we set `c̃ = c + diag(Q)`, form `Q̃ = Q - Diag(diag(Q))`, and eigendecompose `Q̃` to get new factors `Ṽ, Λ̃`. This is necessary because the flip readout `x_i = (σ_i > 0)` relies on the anti-symmetry `σ_{+e_i} = -σ_{-e_i}`, which by Theorem 4 in the paper holds iff `e_i'Qe_i = Q_{ii} = 0`.
- **`test_sweep.jl`** ✅ — Stress test with progress bar for the constrained solver.

### Debug

- **`debug_zerodiag.jl`** — Verifies that flip arrangement predicates are consistent with brute-force optima when `diag(Q) = 0`. Confirms 300/300 across n=5..10.
- **`debug_consistency.jl`** — Demonstrates that the zero-diagonal assumption is essential: evaluates predicates at the brute-force optimal embedding `ξ* = V'x*` and shows inconsistencies when `Q_{ii} ≠ 0`.
- **`debug_unconstrained.jl`** — Inspects IFS sign enumeration for specific seeds: checks whether the ground-truth sign vector appears in the IFS output.

## python/

Earlier Python prototypes organized by rank.

- **`rank1/`** — Rank-1 QUBO solver and tests (`shared.py`, `sweep.py`, `test_stab.py`).
- **`rank2/`** — Rank-2 QUBO solver (`rank2_qubo.py`, `sweep.py`).
- **`rankk/`** — General rank-k solver and tests (`sweep.py`, `test_stab.py`).
