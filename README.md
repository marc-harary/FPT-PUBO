# FPT-PUBO

Fixed-parameter tractable solver for Polynomial Unconstrained Binary Optimization via hyperplane arrangement enumeration.

## julia/

### Core

- **`LowRankQUBO_IFS.jl`** — Main solver module. Provides `qubo_cc` (cardinality-constrained QUBO via swap arrangements and SCC-based readout) and `qubo_unconstrained` (unconstrained QUBO via flip arrangements). Built on `IncSignFeas` for hyperplane arrangement enumeration and `Graphs.jl` for exchange graph analysis.

### Demos

- **`demo.jl`** — Multi-trial test runner for cardinality-constrained QUBO. Runs 80 trials (4 configs x 20 seeds) comparing `qubo_cc` against brute-force enumeration.
- **`demo_unconstrained.jl`** — Multi-trial test runner for unconstrained QUBO. Generates random low-rank Q, zeros the diagonal (absorbing into the linear term), eigendecomposes, and verifies against brute force. Runs 200 trials across n=6,8,10,12.
- **`test_sweep.jl`** — Stress test with progress bar (via `ProgressMeter`) for the constrained solver across multiple (n, r, k) configurations.

### Debug

- **`debug_zerodiag.jl`** — Verifies that the flip arrangement predicates are consistent with the brute-force optimum when Q has zero diagonal. Runs 300 trials across n=5..10.
- **`debug_consistency.jl`** — Inspects whether the brute-force optimal solution is arrangement-consistent by evaluating all predicates at the optimal embedding point.
- **`debug_unconstrained.jl`** — Inspects IFS sign enumeration for failing unconstrained cases: checks whether the ground-truth sign vector appears in the IFS output.
