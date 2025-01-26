# Eigh API
# --------
# TODO: export? or not export but mark as public ?
function eigh!(A::AbstractMatrix, args...; kwargs...)
    return eigh_full!(A, args...; kwargs...)
end

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
# copy input
function copy_input(::typeof(eigh_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(eigh_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

# initialize output
function initialize_output(::typeof(eigh_full!), A::AbstractMatrix, ::LAPACK_EighAlgorithm)
    n = size(A, 1) # square check will happen later
    D = Diagonal(similar(A, real(eltype(A)), n))
    V = similar(A, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eigh_vals!), A::AbstractMatrix, ::LAPACK_EighAlgorithm)
    n = size(A, 1) # square check will happen later
    D = similar(A, real(eltype(A)), n)
    return D
end

# select default algorithm
function select_algorithm(::typeof(eigh_full!), A::AbstractMatrix; kwargs...)
    return default_eigh_algorithm(A; kwargs...)
end
function select_algorithm(::typeof(eigh_vals!), A::AbstractMatrix; kwargs...)
    return default_eigh_algorithm(A; kwargs...)
end

function default_eigh_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_MultipleRelativelyRobustRepresentations(; kwargs...)
end

# check input
function check_input(::typeof(eigh_full!), A::AbstractMatrix, DV)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    D, V = DV
    (V isa AbstractMatrix && eltype(V) == eltype(A) && size(V) == (m, m)) ||
        throw(ArgumentError("`eigh_full!` requires square V matrix with same size and `eltype` as A"))
    (D isa Diagonal && eltype(D) == real(eltype(A)) && size(D) == (m, m)) ||
        throw(ArgumentError("`eigh_full!` requires Diagonal matrix D with same size as A with a real `eltype`"))
    return nothing
end
function check_input(::typeof(eigh_vals!), A::AbstractMatrix, D)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    (size(D) == (n,) && eltype(D) == real(eltype(A))) ||
        throw(ArgumentError("Eigenvalue vector `D` must have length equal to size(A, 1) with a real `eltype`"))
    return nothing
end

# actual implementation
function eigh_full!(A::AbstractMatrix, DV, alg::LAPACK_EighAlgorithm)
    check_input(eigh_full!, A, DV)
    D, V = DV
    Dd = D.diag
    if alg isa LAPACK_MultipleRelativelyRobustRepresentations
        YALAPACK.heevr!(A, Dd, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, Dd, V; alg.kwargs...)
    elseif alg isa LAPACK_Simple
        YALAPACK.heev!(A, Dd, V; alg.kwargs...)
    else # alg isa LAPACK_Expert
        YALAPACK.heevx!(A, Dd, V; alg.kwargs...)
    end
    return D, V
end

function eigh_vals!(A::AbstractMatrix, D, alg::LAPACK_EighAlgorithm)
    check_input(eigh_vals!, A, D)
    V = similar(A, (size(A, 1), 0))
    if alg isa LAPACK_MultipleRelativelyRobustRepresentations
        YALAPACK.heevr!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_QRIteration # == LAPACK_Simple
        YALAPACK.heev!(A, D, V; alg.kwargs...)
    else # alg isa LAPACK_Bisection == LAPACK_Expert
        YALAPACK.heevx!(A, D, V; alg.kwargs...)
    end
    return D
end

function eigh_trunc!(A::AbstractMatrix, DV, alg::TruncatedDenseEig)
    D, V = eigh_full!(A, DV, alg.eig_alg)
    ind = findtruncated(diagview(D), alg.trunc_alg)
    return truncate!((D, V), ind)
end

copy_input(::typeof(eigh_trunc), A) = copy_input(eigh_full, A)

function select_algorithm(::typeof(eigh_trunc!), A::AbstractMatrix; kwargs...)
    return TruncatedDenseEig(default_eigh_algorithm(A; kwargs...), NoTruncation())
end
function initialize_output(::typeof(eigh_trunc!), A::AbstractMatrix, alg::TruncatedDenseEig)
    return initialize_output(eigh_full!, A, alg.eig_alg)
end
