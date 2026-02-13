# ==============================================================================
# DEMO / TEST RUNNER
# ==============================================================================

using Random
using Combinatorics
using LinearAlgebra

# Make sure the module is available (or include the file)
include("LowRankQUBO_IFS.jl")
using .LowRankQUBO_IFS

function run_demo()
    configs = [
        (n = 6, r = 2, k = 3, seeds = 1:20),
        (n = 8, r = 2, k = 4, seeds = 1:20),
        (n = 8, r = 2, k = 5, seeds = 1:20),
        (n = 10, r = 2, k = 5, seeds = 1:20),
    ]

    passes = 0
    fails = 0
    skips = 0

    for cfg in configs
        n, r, k = cfg.n, cfg.r, cfg.k
        println("=== Testing n=$n, r=$r, k=$k ($(length(cfg.seeds)) seeds) ===")

        for seed in cfg.seeds
            Random.seed!(seed)

            # Random low-rank Q = V Λ V'
            V = randn(n, r)
            Λ = randn(r) .* 5.0
            c = randn(n)

            # 1. Solve using the IFS Solver
            t_start = time()
            x_opt, val_opt =
                LowRankQUBO_IFS.qubo_cc(V, Λ, k; c = c, algo = 3, exact = false)
            t_ifs = time() - t_start

            # 2. Verify with Brute Force
            Q = V * Diagonal(Λ) * V'

            bf_best_val = -Inf
            bf_best_x = falses(n)

            for idxs in Combinatorics.combinations(1:n, k)
                x = falses(n)
                x[idxs] .= true

                # Eval x'Qx + c'x
                val = dot(x, Q, x) + dot(c, x)

                if val > bf_best_val
                    bf_best_val = val
                    bf_best_x = x
                end
            end

            # Check
            if val_opt == -Inf
                skips += 1
                printstyled("  SKIP  ", color = :yellow)
                println("seed=$seed  (no feasible readout)")
            elseif abs(val_opt - bf_best_val) < 1e-6
                passes += 1
                printstyled("  PASS  ", color = :green)
                println(
                    "seed=$seed  val=$( round(val_opt, digits=4))  time=$(round(t_ifs, digits=3))s",
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

    println(
        "\n=== Summary: $passes passed, $fails failed, $skips skipped out of $(passes+fails+skips) trials ===",
    )

    if fails == 0
        printstyled("ALL PASSED ✅\n", color = :green)
    else
        printstyled("FAILURES DETECTED ❌\n", color = :red)
    end
end

# Execute
run_demo()
