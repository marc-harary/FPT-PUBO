# Quick debug: check arrangement consistency for bipartite zero-diag construction

using Random, LinearAlgebra
include("LowRankQUBO_IFS.jl")
using .LowRankQUBO_IFS

function debug_exact(seed; n = 6, r = 1)
    Random.seed!(seed)
    B = randn(r, n - r)
    Q = zeros(n, n)
    Q[1:r, (r+1):n] = B
    Q[(r+1):n, 1:r] = B'
    c = randn(n)

    eig = eigen(Symmetric(Q))
    nz = findall(abs.(eig.values) .> 1e-10)
    V = eig.vectors[:, nz]
    Λ = eig.values[nz]

    println("diag(Q) = ", diag(Q))
    println("diag(V*Diag(Λ)*V') = ", round.(diag(V * Diagonal(Λ) * V'), digits = 15))

    A, t = LowRankQUBO_IFS.flip_arrangement(V, Λ, c)

    # BF
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
    println("\nBF optimal: x* = ", Int.(bf_best_x), "  val = ", bf_best_val)

    ξ = V' * Float64.(bf_best_x)
    println("ξ* = ", round.(ξ, digits = 6))

    consistent = true
    for i = 1:n
        val = dot(A[:, i], ξ) + t[i]
        expected = bf_best_x[i] ? 1 : -1
        sign_i = val > 0 ? 1 : (val < 0 ? -1 : 0)
        ok = sign_i == expected
        if !ok
            consistent = false
        end
        println(
            "  h_$i: val=$(round(val, digits=8))  sign=$sign_i  x*[$i]=$(Int(bf_best_x[i]))  $(ok ? "ok" : "INCONSISTENT")",
        )
    end

    # Also print what Q_ii + c_i looks like (the intercept)
    println("\nIntercept check:")
    for i = 1:n
        qii = sum(Λ[u] * V[i, u]^2 for u = 1:length(Λ))
        println(
            "  i=$i: Q_{ii}=$(round(qii, digits=10))  c_i=$(round(c[i], digits=6))  t_i=$(round(t[i], digits=6))",
        )
    end

    return consistent
end

debug_exact(1; n = 6, r = 1)
