# ==============================================================================
# DEBUG: Check if brute-force optimal is arrangement-consistent
# ==============================================================================

using Random
using LinearAlgebra

include("LowRankQUBO_IFS.jl")
using .LowRankQUBO_IFS

function brute_force_unconstrained(V, Λ, c)
    n = size(V, 1)
    Q = V * Diagonal(Λ) * V'

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

function debug_seed(seed; n = 6, r = 2)
    Random.seed!(seed)
    V = randn(n, r)
    Λ = randn(r) .* 5.0
    c = randn(n)

    println("\n" * "="^60)
    println("SEED $seed  (n=$n, r=$r)")
    println("="^60)

    x_bf, v_bf = brute_force_unconstrained(V, Λ, c)
    println("BF optimal: x* = $(Int.(x_bf))  val = $v_bf")

    # Build flip arrangement
    A, t = LowRankQUBO_IFS.flip_arrangement(V, Λ, c)

    # ξ* = V' x*
    ξ_star = V' * Float64.(x_bf)
    println("ξ* = V' x* = $(round.(ξ_star, digits=6))")

    # Evaluate each predicate at ξ*
    println("\nPredicate evaluation at ξ*:")
    gt_sign = zeros(Int, n)
    for i = 1:n
        val = dot(A[:, i], ξ_star) + t[i]
        gt_sign[i] = val > 0 ? 1 : (val < 0 ? -1 : 0)
        println(
            "  h_$i: a'ξ* + t = $(round(val, digits=8))  sign=$(gt_sign[i])  x*[$i]=$(Int(x_bf[i]))",
        )
    end

    # Check consistency: does sign agree with x*?
    # If x[i]=1, we expect the flip 0→1 was beneficial, so predicate should be positive
    # If x[i]=0, we expect the flip 0→1 was NOT beneficial, so predicate should be negative
    # But wait — the predicate measures the marginal gain of flipping x[i] at the CURRENT ξ,
    # not at the ξ where x[i]=0.
    println("\nConsistency check (sign vs x*):")
    consistent = true
    for i = 1:n
        expected = x_bf[i] ? 1 : -1
        ok = (gt_sign[i] == expected)
        if !ok
            consistent = false
            printstyled("  INCONSISTENT ", color = :red)
            println("i=$i: x*[$i]=$(Int(x_bf[i])) but sign=$(gt_sign[i])")
        end
    end

    if consistent
        printstyled("  All consistent ✅\n", color = :green)
    else
        # Show what the "sign-consistent" x would be
        x_from_sign = falses(n)
        for i = 1:n
            x_from_sign[i] = (gt_sign[i] > 0)
        end
        v_from_sign = LowRankQUBO_IFS.obj(V, Λ, c, x_from_sign)
        println("\n  Sign-implied x = $(Int.(x_from_sign))  val = $v_from_sign")
        println("  BF optimal   x = $(Int.(x_bf))           val = $v_bf")
        println("  Gap: $(v_bf - v_from_sign)")

        # Also check: is x* a local optimum? (no single flip improves it)
        Q = V * Diagonal(Λ) * V'
        println("\n  Single-flip analysis from x*:")
        is_local_opt = true
        for i = 1:n
            x_flip = copy(x_bf)
            x_flip[i] = !x_flip[i]
            v_flip = dot(x_flip, Q, x_flip) + dot(c, x_flip)
            delta = v_flip - v_bf
            if delta > 1e-10
                is_local_opt = false
                printstyled(
                    "    flip x[$i]: Δ = $(round(delta, digits=6)) (IMPROVES!) \n",
                    color = :red,
                )
            else
                println("    flip x[$i]: Δ = $(round(delta, digits=6))")
            end
        end
        if is_local_opt
            printstyled("  x* IS a local optimum (swap-stable)\n", color = :green)
        end
    end
end

# Test failing seeds
for seed in [14, 16]
    debug_seed(seed; n = 6, r = 2)
end

println("\n\n--- n=8 failures ---")
for seed in [5, 13, 19]
    debug_seed(seed; n = 8, r = 2)
end
