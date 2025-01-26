# Eigh functions
# --------------

docs_eigh_note = """
Note that `eigh` and its variants assume additional structure on the input,
and therefore will retain the `eltype` of the input for the eigenvalues and eigenvectors.
For generic eigenvalue decompositions, see [`eig`](@ref).
"""

# TODO: do we need "full"?
"""
    eigh_full(A; kwargs...) -> D, V
    eigh_full(A, alg::AbstractAlgorithm) -> D, V
    eigh_full!(A, [DV]; kwargs...) -> D, V
    eigh_full!(A, [DV], alg::AbstractAlgorithm) -> D, V

Compute the symmetric or hermitian eigenvalue decomposition of `A`
such that `A * V = V * D`.

$(docs_eigh_note)

See also [`eigh_vals(!)`](@ref eigh_vals) and [`eigh_trunc(!)`](@ref).
"""
@functiondef eigh_full

"""
    eigh_trunc(A; kwargs...) -> D, V
    eigh_trunc(A, alg::AbstractAlgorithm) -> D, V
    eigh_trunc!(A, [DV]; kwargs...) -> D, V
    eigh_trunc!(A, [DV], alg::AbstractAlgorithm) -> D, V


Compute the symmetric or hermitian truncated eigenvalue decomposition of `A`
such that `A * V ≈ V * D`.

$(docs_eigh_note)

See also [`eigh_full(!)`](@ref eigh_full) and [`eigh_vals(!)`](@ref eigh_vals).
"""
@functiondef eigh_trunc

"""
    eigh_vals(A; kwargs...) -> D
    eigh_vals(A, alg::AbstractAlgorithm) -> D
    eigh_vals!(A, [D]; kwargs...) -> D
    eigh_vals!(A, [D], alg::AbstractAlgorithm) -> D

Compute the vector of (real) eigenvalues of symmetric or hermitian `A`.

$(docs_eigh_note)

See also [`eigh_full(!)`](@ref eigh_full) and [`eigh_trunc(!)`](@ref eigh_trunc).
"""
@functiondef eigh_vals

# Algorithm selection
# -------------------
for f in (:eigh_full, :eigh_vals)
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

function select_algorithm(::typeof(eigh_trunc), A; kwargs...)
    return select_algorithm(eigh_trunc!, A; kwargs...)
end
function select_algorithm(::typeof(eigh_trunc!), A; alg=nothing, trunc=nothing, kwargs...)
    alg_eigh = select_algorithm(eigh_full!, A; alg, kwargs...)
    alg_trunc = trunc isa TruncationStrategy ? trunc :
                trunc isa NamedTuple ? TruncationStrategy(; trunc...) :
                isnothing(trunc) ? NoTruncation() :
                throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return TruncatedAlgorithm(alg_eigh, alg_trunc)
end

# Default to LAPACK 
function default_eigh_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_MultipleRelativelyRobustRepresentations(; kwargs...)
end
