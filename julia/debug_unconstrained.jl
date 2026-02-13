# ==============================================================================
# DEBUG: Inspect IFS sign enumeration for failing unconstrained cases
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

    # --- Brute force optimal ---
    x_bf, v_bf = brute_force_unconstrained(V, Λ, c)
    println("\nBrute force optimal:")
    println("  x* = $(Int.(x_bf))")
    println("  val = $v_bf")

    # --- Build flip arrangement ---
    A, t = LowRankQUBO_IFS.flip_arrangement(V, Λ, c)

    println("\nFlip arrangement (A columns = normal vectors, t = intercepts):")
    for i = 1:n
        println("  h_$i: a=$(round.(A[:,i], digits=4))  t=$(round(t[i], digits=4))")
    end

    # --- Ground-truth sign at ξ* = V' x* ---
    ξ_star = V' * Float64.(x_bf)
    println("\nξ* = V' x* = $(round.(ξ_star, digits=4))")

    gt_sign = zeros(Int, n)
    for i = 1:n
        val = dot(A[:, i], ξ_star) + t[i]
        gt_sign[i] = val > 0 ? 1 : (val < 0 ? -1 : 0)
        println(
            "  predicate $i: a'ξ* + t = $(round(val, digits=6))  →  sign = $(gt_sign[i])",
        )
    end
    println("\nGround-truth sign vector: $gt_sign")

    # --- IFS enumeration ---
    sigs = LowRankQUBO_IFS.chamber_signs(A, t; algo = 3, exact = false)

    # Normalize to list of vectors
    sig_list = if sigs isa AbstractDict
        collect(keys(sigs))
    elseif sigs isa AbstractMatrix
        [sigs[:, j] for j = 1:size(sigs, 2)]
    else
        collect(sigs)
    end

    println("\nIFS returned $(length(sig_list)) sign vectors:")
    found_gt = false
    for (idx, s) in enumerate(sig_list)
        sv = [s[i] > 0 ? 1 : (s[i] < 0 ? -1 : 0) for i = 1:length(s)]

        # Readout and eval
        x = falses(n)
        for i = 1:n
            x[i] = (s[i] > 0)
        end
        v = LowRankQUBO_IFS.obj(V, Λ, c, x)

        match = (sv == gt_sign)
        if match
            found_gt = true
        end

        marker = match ? " ← GT" : ""
        println("  [$idx] sign=$sv  x=$(Int.(x))  val=$(round(v, digits=4))$marker")
    end

    if found_gt
        printstyled(
            "\n✅ Ground-truth sign vector WAS found in IFS output\n",
            color = :green,
        )
    else
        printstyled("\n❌ Ground-truth sign vector NOT found in IFS output\n", color = :red)

        # Check if gt_sign has any zeros (degenerate case)
        if any(gt_sign .== 0)
            printstyled(
                "  ⚠ Ground-truth sign has zeros (point lies on hyperplane)\n",
                color = :yellow,
            )
        end
    end
end

# Debug the failing seeds
for seed in [14, 16, 19]
    debug_seed(seed)
end
