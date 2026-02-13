# Multi-trial stress test: compare IFS solver vs brute force
using Random, Combinatorics, LinearAlgebra
include("LowRankQUBO_IFS.jl")
using .LowRankQUBO_IFS

function brute_force(V, Λ, c, k)
    n = size(V, 1)
    Q = V * Diagonal(Λ) * V'
    best_val = -Inf
    best_x = falses(n)
    for idxs in combinations(1:n, k)
        x = falses(n)
        x[idxs] .= true
        val = dot(x, Q, x) + dot(c, x)
        if val > best_val
            best_val = val
            best_x = x
        end
    end
    return best_x, best_val
end

function run_sweep()
    passes = 0
    fails = 0
    skips = 0

    for seed in 1:20
        for (n, r, k) in [(6,2,3), (8,2,4), (8,2,5), (10,2,5)]
            Random.seed!(seed)
            V = randn(n, r)
            Λ = randn(r) .* 5.0
            c = randn(n)

            x_ifs, v_ifs = LowRankQUBO_IFS.qubo_cc(V, Λ, k; c=c, algo=3, exact=false)
            _, v_bf = brute_force(V, Λ, c, k)

            if v_ifs == -Inf
                skips += 1
                printstyled("  SKIP  ", color=:yellow)
                println("seed=$seed n=$n r=$r k=$k  (no feasible readout)")
            elseif abs(v_ifs - v_bf) < 1e-6
                passes += 1
            else
                fails += 1
                printstyled("  FAIL  ", color=:red)
                println("seed=$seed n=$n r=$r k=$k  IFS=$v_ifs  BF=$v_bf  gap=$(v_bf - v_ifs)")
            end
        end
    end

    println("\n=== Summary: $passes passed, $fails failed, $skips skipped out of $(passes+fails+skips) trials ===")
    fails == 0 ? printstyled("ALL PASSED ✅\n", color=:green) : printstyled("FAILURES DETECTED ❌\n", color=:red)
end

run_sweep()
