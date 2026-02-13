# ==============================================================================
# DEBUG: Generate random Q with zero diagonal, eigendecompose, check predicates
# ==============================================================================

using Random
using LinearAlgebra

include("LowRankQUBO_IFS.jl")
using .LowRankQUBO_IFS

function brute_force_unconstrained(Q, c)
    n = size(Q, 1)
    bf_best_val = 0.0
    bf_best_x = falses(n)

    for mask = 1:(2^n-1)
        x = falses(n)
        for i = 1:n
            x[i] = (mask >> (i - 1)) & 1 == 1
        end
        val = dot(x, Q, x) + dot(c, x)
        if val > bf_best_val
            bf_best_val = val
            bf_best_x = x
        end
    end

    return bf_best_x, bf_best_val
end

function debug_seed(seed; n = 5)
    Random.seed!(seed)

    # Random symmetric matrix, then zero the diagonal
    M = randn(n, n)
    Q = (M + M') / 2
    Q[diagind(Q)] .= 0.0

    c = randn(n)

    # Eigendecompose: Q = V Λ V'
    eig = eigen(Symmetric(Q))
    Λ = eig.values
    V = eig.vectors

    r = length(Λ)  # full rank (almost surely)

    println("\n" * "="^60)
    println("SEED $seed  (n=$n, r=$r)")
    println("="^60)
    println("Q =")
    display(round.(Q, digits = 4))
    println("\nΛ = $(round.(Λ, digits=4))")
    println("diag(Q) = $(round.(diag(Q), digits=10))")

    # Brute force
    x_bf, v_bf = brute_force_unconstrained(Q, c)
    println("\nBF optimal: x* = $(Int.(x_bf))  val = $v_bf")

    # Build flip arrangement from V, Λ
    A, t = LowRankQUBO_IFS.flip_arrangement(V, Λ, c)

    # ξ* = V' x*
    ξ_star = V' * Float64.(x_bf)
    println("ξ* = V' x* = $(round.(ξ_star, digits=6))")

    # Evaluate each predicate at ξ*
    println("\nPredicate evaluation at ξ*:")
    consistent = true
    for i = 1:n
        val = dot(A[:, i], ξ_star) + t[i]
        sign_i = val > 0 ? 1 : (val < 0 ? -1 : 0)
        expected = x_bf[i] ? 1 : -1
        ok = (sign_i == expected)
        if !ok
            consistent = false
            printstyled("  INCONSISTENT ", color = :red)
        else
            printstyled("  ok           ", color = :green)
        end
        println(
            "i=$i: a'ξ*+t = $(round(val, digits=8))  sign=$sign_i  x*[$i]=$(Int(x_bf[i]))",
        )
    end

    if consistent
        printstyled("  All consistent ✅\n", color = :green)
    else
        printstyled("  INCONSISTENCY FOUND ❌\n", color = :red)
    end

    return consistent
end

function run_all()
    pass = 0
    fail = 0
    for n in [5, 6, 7, 8, 9, 10]
        for seed = 1:50
            ok = debug_seed(seed; n = n)
            ok ? (pass += 1) : (fail += 1)
        end
    end

    println("\n\n" * "="^60)
    println("OVERALL: $pass consistent, $fail inconsistent out of $(pass+fail)")
    if fail == 0
        printstyled("ALL CONSISTENT ✅\n", color = :green)
    else
        printstyled("INCONSISTENCIES FOUND ❌\n", color = :red)
    end
end

run_all()
