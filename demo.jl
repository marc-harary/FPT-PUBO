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
    Random.seed!(42)

    # Setup Problem: n=10, r=2, k=5
    n, r, k = 8, 2, 5
    println("Generating Rank-$r Problem (n=$n, k=$k)...")

    # Random low-rank Q = V Λ V'
    V = randn(n, r)
    # Random eigenvalues (mixed signs to ensure non-convexity)
    Λ = [10.0, -5.0]
    c = randn(n)

    # 1. Solve using the IFS Solver
    println("\nRunning LowRankQUBO_IFS...")
    t_start = time()
    x_opt, val_opt = LowRankQUBO_IFS.qubo_cc(V, Λ, k; c = c, algo = 3, exact = false)
    t_end = time()

    println("  Optimal Value: $val_opt")
    println("  Support:       $(findall(x_opt))")
    println("  Time:          $(round(t_end - t_start, digits=4))s")

    # 2. Verify with Brute Force (Iterating all binom(10,5) = 252 combinations)
    println("\nVerifying with Brute Force...")

    # Reconstruct Q for checking
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

    println("  Brute Force:   $bf_best_val")

    # Check
    if abs(val_opt - bf_best_val) < 1e-6
        println("\n✅ MATCH CONFIRMED")
    else
        println("\n❌ MISMATCH")
        println("  IFS found: $val_opt")
        println("  BF  found: $bf_best_val")
    end
end

# Execute
run_demo()
