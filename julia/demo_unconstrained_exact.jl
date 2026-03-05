# ==============================================================================
# DEMO / TEST RUNNER — Unconstrained QUBO with exact zero-diagonal construction
# ==============================================================================
#
# Constructs rank-2r matrices with exactly zero diagonal using the bipartite
# off-diagonal block trick:
#
#   Q = [  0    B  ]
#       [ B'   0   ]
#
# where B ∈ ℝ^{r×(n-r)} has row-rank r. Then rank(Q) = 2r and diag(Q) = 0.
# We eigendecompose Q to get V, Λ and feed directly into the solver.
# ==============================================================================

using Random
using LinearAlgebra

include("LowRankQUBO_IFS.jl")
using .LowRankQUBO_IFS

function make_zerodiag_rank2r(n::Int, r::Int)
    @assert 2r <= n "Need 2r ≤ n for this construction"
    B = randn(r, n - r)
    Q = zeros(n, n)
    Q[1:r, (r+1):n] = B
    Q[(r+1):n, 1:r] = B'
    return Q
end

function brute_force_shifted(Q, c, mshift)
    n = size(Q, 1)
    bf_best_val = 0.0
    bf_best_x = falses(n)

    for mask = 1:(2^n-1)
        x = falses(n)
        for i = 1:n
            x[i] = (mask >> (i - 1)) & 1 == 1
        end

        k = count(x)
        val = dot(x, Q, x) + dot(c, x) + mshift * k^2

        if val > bf_best_val
            bf_best_val = val
            bf_best_x = x
        end
    end

    return bf_best_x, bf_best_val
end

function run_demo()
    configs = [
        (n = 6, r = 1, seeds = 1:100),   # rank 2
        (n = 8, r = 1, seeds = 1:100),   # rank 2
        (n = 8, r = 2, seeds = 1:100),   # rank 4
        (n = 10, r = 1, seeds = 1:100),  # rank 2
        (n = 10, r = 2, seeds = 1:100),  # rank 4
        (n = 12, r = 1, seeds = 1:100),  # rank 2
        (n = 12, r = 2, seeds = 1:100),  # rank 4
        (n = 12, r = 3, seeds = 1:100),  # rank 6
    ]

    passes = 0
    fails = 0
    total = sum(length(cfg.seeds) for cfg in configs)

    for cfg in configs
        n, r = cfg.n, cfg.r
        println("=== Testing n=$n, rank=2×$r=$(2r) ($(length(cfg.seeds)) seeds) ===")

        for seed in cfg.seeds
            Random.seed!(seed)

            # Construct exact zero-diagonal rank-2r matrix
            Q = make_zerodiag_rank2r(n, r)
            c = randn(n)

            # Verify construction
            @assert maximum(abs.(diag(Q))) < 1e-15 "diag(Q) ≠ 0!"
            actual_rank = rank(Q)
            @assert actual_rank == 2r "Expected rank $(2r), got $actual_rank"

            # Eigendecompose to get V, Λ — keep only the 2r nonzero eigenvalues
            eig = eigen(Symmetric(Q))
            nonzero = findall(abs.(eig.values) .> 1e-10)
            @assert length(nonzero) == 2r "Expected 2r=$( 2r) nonzero eigenvalues, got $(length(nonzero))"
            V = eig.vectors[:, nonzero]
            Λ = eig.values[nonzero]

            # Compute mshift = -min_i Q_{ii} for brute force comparison
            qdiag = [sum(Λ[u] * V[i, u]^2 for u = 1:length(Λ)) for i = 1:n]
            mshift = -minimum(qdiag)

            # Solve using IFS
            t_start = time()
            x_opt, val_opt =
                LowRankQUBO_IFS.qubo_unconstrained(V, Λ; c = c, algo = 3, exact = false)
            t_ifs = time() - t_start

            # Verify with brute force (same shifted objective)
            _, bf_best_val = brute_force_shifted(Q, c, mshift)

            # Check
            if abs(val_opt - bf_best_val) < 1e-6
                passes += 1
                printstyled("  PASS  ", color = :green)
                println(
                    "seed=$seed  val=$(round(val_opt, digits=4))  |x|=$(count(x_opt))  time=$(round(t_ifs, digits=3))s",
                )
            else
                fails += 1
                printstyled("  FAIL  ", color = :red)
                println(
                    "seed=$seed  IFS=$val_opt  BF=$bf_best_val  gap=$(bf_best_val - val_opt)",
                )
            end
        end
    end

    println("\n=== Summary: $passes passed, $fails failed out of $total trials ===")

    if fails == 0
        printstyled("ALL PASSED ✅\n", color = :green)
    else
        printstyled("FAILURES DETECTED ❌\n", color = :red)
    end
end

# Execute
run_demo()
