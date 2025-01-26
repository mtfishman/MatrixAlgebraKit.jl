# Eig functions
# -------------

# TODO: kwargs for sorting eigenvalues?

docs_eig_note = """
Note that `eig` and its variants do not assume additional structure on the input,
and therefore will always return complex eigenvalues and eigenvectors. For real
eigenvalue decompositions of symmetric or hermitian matrices, see [`eigh`](@ref).
"""

# TODO: do we need "full"?
"""
    eig_full(A; kwargs...) -> D, V
    eig_full(A, alg::AbstractAlgorithm) -> D, V
    eig_full!(A, [DV]; kwargs...) -> D, V
    eig_full!(A, [DV], alg::AbstractAlgorithm) -> D, V

Compute the eigenvalue decomposition of `A` such that `A * V = V * D`.

$(docs_eig_note)

See also [`eig_vals(!)`](@ref eig_vals) and [`eig_trunc(!)`](@ref).
"""
@functiondef eig_full

"""
    eig_trunc(A; kwargs...) -> D, V
    eig_trunc(A, alg::AbstractAlgorithm) -> D, V
    eig_trunc!(A, [DV]; kwargs...) -> D, V
    eig_trunc!(A, [DV], alg::AbstractAlgorithm) -> D, V


Compute the truncated eigenvalue decomposition of `A` such that `A * V ≈ V * D`.

$(docs_eig_note)

See also [`eig_full(!)`](@ref eig_full) and [`eig_vals(!)`](@ref eig_vals).
"""
@functiondef eig_trunc

"""
    eig_vals(A; kwargs...) -> D
    eig_vals(A, alg::AbstractAlgorithm) -> D
    eig_vals!(A, [D]; kwargs...) -> D
    eig_vals!(A, [D], alg::AbstractAlgorithm) -> D

Compute the vector of eigenvalues of `A`.

$(docs_eig_note)

See also [`eig_full(!)`](@ref eig_full) and [`eig_trunc(!)`](@ref eig_trunc).
"""
@functiondef eig_vals

# Algorithm selection
# -------------------
for f in (:eig_full, :eig_vals)
    f! = Symbol(f, :!)
    @eval begin
        function select_algorithm(::typeof($f), A; kwargs...)
            return select_algorithm($f!, A; kwargs...)
        end
        function select_algorithm(::typeof($f!), A; alg=nothing, kwargs...)
            if alg isa AbstractAlgorithm
                return alg
            elseif alg isa Symbol
                return Algorithm{alg}(; kwargs...)
            else
                isnothing(alg) || throw(ArgumentError("Unknown alg $alg"))
                return default_eig_algorithm(A; kwargs...)
            end
        end
    end
end

function select_algorithm(::typeof(eig_trunc), A; kwargs...)
    return select_algorithm(eig_trunc!, A; kwargs...)
end
function select_algorithm(::typeof(eig_trunc!), A; alg=nothing, trunc=nothing, kwargs...)
    alg_eig = select_algorithm(eig_full!, A; alg, kwargs...)
    alg_trunc = trunc isa TruncationStrategy ? trunc :
                trunc isa NamedTuple ? TruncationStrategy(; trunc...) :
                isnothing(trunc) ? NoTruncation() :
                throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return TruncatedAlgorithm(alg_eig, alg_trunc)
end

# Default to LAPACK 
function default_eig_algorithm(A::StridedMatrix{<:BlasFloat}; kwargs...)
    return LAPACK_Expert(; kwargs...)
end
