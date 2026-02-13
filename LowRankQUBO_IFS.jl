module LowRankQUBO_IFS

using LinearAlgebra
using IncSignFeas
using Graphs

# =========================
# Low-rank model: Q = V*Diagonal(Λ)*V'  (V: n×r, Λ: r)
# We work in **eigen-coordinates** ξ := V' x ∈ ℝ^r, so no B,S conversion.
# For swaps d = e_i - e_j (i≠j):
#   Δ_{i←j}(ξ) = a_{ij}' ξ + b_{ij}
# where
#   a_{ij} = 2*Diagonal(Λ)*(V[i,:] - V[j,:])
#   b_{ij} = (c_i - c_j) + d'Qd, and with diag(Q)=0: d'Qd = -2Q_ij
#   Q_ij = V[i,:]'*Diagonal(Λ)*V[j,:]
# =========================

@inline function qij(V, Λ, i::Int, j::Int)
    s = 0.0
    @inbounds for t = 1:length(Λ)
        s += V[i, t] * Λ[t] * V[j, t]
    end
    return s
end

# Build arrangement (A,t) and predicate index ↔ (i,j) map, directly from (V,Λ)
function swap_arrangement(V::AbstractMatrix, Λ::AbstractVector, c::AbstractVector)
    n, r = size(V)
    @assert length(Λ) == r
    @assert length(c) == n

    m = n*(n-1)
    A = Matrix{Float64}(undef, r, m)   # columns are a_{ij}
    t = Vector{Float64}(undef, m)      # intercepts b_{ij}
    pairs = Vector{NTuple{2,Int}}(undef, m)

    k = 0
    @inbounds for i = 1:n, j = 1:n
        i==j && continue
        k += 1
        pairs[k] = (i, j)

        # a_{ij} = 2*Λ ⊙ (V_i - V_j)
        for u = 1:r
            A[u, k] = 2.0 * Λ[u] * (V[i, u] - V[j, u])
        end

        # b_{ij} = (c_i-c_j) - 2 Q_ij   (assuming diag(Q)=0 absorbed into c)
        t[k] = (c[i] - c[j]) - 2.0*qij(V, Λ, i, j)
    end

    return A, t, pairs
end

# IFS chamber enumeration: return sign vectors container (IncSignFeas contract)
function chamber_signs(A, t; algo::Int = 3, exact::Bool = false)
    H = vcat(A, reshape(t, 1, :))                    # per README: [V ; t'] style packing
    info = isf(H, options_from_algo(algo, exact))
    return info.s
end

# Build exchange graph directly from sign vector (no ξ).
# Convention: s[p] > 0 means Δ_{i←j} > 0, hence edge j → i.
function graph_from_signs(n::Int, pairs::Vector{NTuple{2,Int}}, s)
    g = SimpleDiGraph(n)
    @inbounds for p = 1:length(pairs)
        (i, j) = pairs[p]
        s[p] > 0 && add_edge!(g, j, i)
    end
    return g
end

# k-closure readout using Graphs.jl SCC + condensation DAG topo.
# Returns sorted support S (|S|=k) or nothing if ambiguous (SCC split needed).
function readout_kclosure(g::SimpleDiGraph, k::Int)
    sccs = strongly_connected_components(g)

    # condensation DAG of SCCs
    cg = condensation(g, sccs)                 # vertices = SCC ids
    topo = topological_sort_by_dfs(cg)         # sources -> sinks

    S = Int[]
    for cid in reverse(topo)                   # sinks -> sources
        block = sccs[cid]
        (length(S) + length(block) <= k) || return nothing
        append!(S, block)
        length(S) == k && return sort(S)
    end
    return nothing
end

# Objective without forming Q: x'Qx + c'x, where Q=V*Diag(Λ)*V'
function obj(V, Λ, c, x::BitVector)
    n, r = size(V)
    @assert length(Λ) == r
    @assert length(c) == n
    @assert length(x) == n

    z = zeros(Float64, r)                           # z = V' x
    @inbounds for u = 1:r, i = 1:n
        x[i] && (z[u] += V[i, u])
    end

    qxx = 0.0
    @inbounds for u = 1:r
        qxx += Λ[u] * (z[u]*z[u])
    end

    return qxx + dot(c, Float64.(x))
end

# =========================
# Public APIs
# =========================

function qubo_cc(
    V::AbstractMatrix,
    Λ::AbstractVector,
    k::Int;
    c = zeros(Float64, size(V, 1)),
    algo::Int = 3,
    exact::Bool = false,
)
    n, _ = size(V)
    @assert 0 ≤ k ≤ n

    A, t, pairs = swap_arrangement(V, Λ, c)
    sigs = chamber_signs(A, t; algo = algo, exact = exact)

    bestx = falses(n);
    bestv = -Inf

    # normalize sigs: Dict keys, matrix columns, or vector-of-vectors
    iter =
        sigs isa AbstractDict ? keys(sigs) :
        sigs isa AbstractMatrix ? (view(sigs, :, j) for j = 1:size(sigs, 2)) : sigs

    for s in iter
        g = graph_from_signs(n, pairs, s)
        Sset = readout_kclosure(g, k)
        Sset === nothing && continue

        x = falses(n);
        @inbounds for i in Sset
            ;
            x[i] = true;
        end
        v = obj(V, Λ, c, x)
        if v > bestv
            bestv = v
            bestx .= x
        end
    end

    return bestx, bestv
end

function qubo_unconstrained(
    V::AbstractMatrix,
    Λ::AbstractVector;
    c = zeros(Float64, size(V, 1)),
    algo::Int = 3,
    exact::Bool = false,
)
    n, _ = size(V)
    bestx = falses(n);
    bestv = -Inf
    for k = 0:n
        x, v = qubo_cc(V, Λ, k; c = c, algo = algo, exact = exact)
        if v > bestv
            bestv = v
            bestx .= x
        end
    end
    return bestx, bestv
end

end # module
