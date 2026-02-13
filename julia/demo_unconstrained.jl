# ==============================================================================
# DEMO / TEST RUNNER — Unconstrained QUBO
# ==============================================================================

using Random
using LinearAlgebra

# Make sure the module is available (or include the file)
include("LowRankQUBO_IFS.jl")
using .LowRankQUBO_IFS

function brute_force_unconstrained(Q, c_tilde)
    n = size(Q, 1)
    bf_best_val = 0.0
    bf_best_x = falses(n)

    for mask = 1:(2^n-1)
        x = falses(n)
        for i = 1:n
            x[i] = (mask >> (i - 1)) & 1 == 1
        end

        val = dot(x, Q, x) + dot(c_tilde, x)

        if val > bf_best_val
            bf_best_val = val
            bf_best_x = x
        end
    end

    return bf_best_x, bf_best_val
end

function run_demo()
    configs = [
        (n = 6, r = 2, seeds = 1:50),
        (n = 8, r = 2, seeds = 1:50),
        (n = 10, r = 2, seeds = 1:50),
        (n = 12, r = 2, seeds = 1:50),
    ]

    passes = 0
    fails = 0

    for cfg in configs
        n, r = cfg.n, cfg.r
        println("=== Testing unconstrained n=$n, r=$r ($(length(cfg.seeds)) seeds) ===")

        for seed in cfg.seeds
            Random.seed!(seed)

            # Random low-rank Q = V Λ V'
            V = randn(n, r)
            Λ = randn(r) .* 5.0

            # Q with nonzero diagonal
            Q = V * Diagonal(Λ) * V'

            # Zero the diagonal: Q̃ = Q - Diag(diag(Q)), c̃ = diag(Q)
            c_tilde = diag(Q)    # absorb diagonal into linear term (c = 0)
            Q_tilde = Q - Diagonal(c_tilde)

            # Eigendecompose the zero-diagonal Q̃
            eig = eigen(Symmetric(Q_tilde))
            V_tilde = eig.vectors
            Λ_tilde = eig.values

            # 1. Solve using the IFS Solver with the zero-diagonal decomposition
            t_start = time()
            x_opt, val_opt = LowRankQUBO_IFS.qubo_unconstrained(
                V_tilde,
                Λ_tilde;
                c = c_tilde,
                algo = 3,
                exact = false,
            )
            t_ifs = time() - t_start

            # 2. Verify with Brute Force (using original Q and c̃)
            _, bf_best_val = brute_force_unconstrained(Q_tilde, c_tilde)

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

    println(
        "\n=== Summary: $passes passed, $fails failed out of $(passes+fails) trials ===",
    )

    if fails == 0
        printstyled("ALL PASSED ✅\n", color = :green)
    else
        printstyled("FAILURES DETECTED ❌\n", color = :red)
    end
end

# Execute
run_demo()
