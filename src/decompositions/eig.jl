# Eig API
# -------
# TODO: export? or not export but mark as public ?
function eig!(A::AbstractMatrix, args...; kwargs...)
    return eig_full!(A, args...; kwargs...)
end

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

# copy input
function copy_input(::typeof(eig_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(eig_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

# initialize output
function initialize_output(::typeof(eig_full!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = Diagonal(similar(A, Tc, n))
    V = similar(A, Tc, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eig_vals!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = similar(A, Tc, n)
    return D
end

# select default algorithm
function select_algorithm(::typeof(eig_full!), A::AbstractMatrix; kwargs...)
    return default_eig_algorithm(A; kwargs...)
end
function select_algorithm(::typeof(eig_vals!), A::AbstractMatrix; kwargs...)
    return default_eig_algorithm(A; kwargs...)
end

function default_eig_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_Expert(; kwargs...)
end

# check input
function check_input(::typeof(eig_full!), A::AbstractMatrix, DV)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decomposition requires square input matrix"))
    D, V = DV
    Tc = complex(eltype(A))
    (V isa AbstractMatrix && eltype(V) == Tc && size(V) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires square matrix V with same size as A and complex `eltype`"))
    (D isa Diagonal && eltype(D) == Tc && size(D) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires Diagonal matrix D with same size as A and complex `eltype`"))
    return nothing
end
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decomposition requires square input matrix"))
    Tc = complex(eltype(A))
    size(D) == (n,) && eltype(D) == Tc ||
        throw(ArgumentError("Eigenvalue vector `D` must have length equal to size(A, 1) and complex `eltype`"))
    return nothing
end

# actual implementation
function eig_full!(A::AbstractMatrix, DV, alg::LAPACK_EigAlgorithm)
    check_input(eig_full!, A, DV)
    D, V = DV
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple does not accept any keyword arguments"))
        YALAPACK.geev!(A, D.diag, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D.diag, V; alg.kwargs...)
    end
    return D, V
end
function eig_vals!(A::AbstractMatrix, D, alg::LAPACK_EigAlgorithm)
    check_input(eig_vals!, A, D)
    V = similar(A, complex(eltype(A)), (size(A, 1), 0))
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple does not accept any keyword arguments"))
        YALAPACK.geev!(A, D, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D, V; alg.kwargs...)
    end
    return D
end

function eig_trunc!(A::AbstractMatrix, DV, alg::TruncatedAlgorithm)
    D, V = eig_full!(A, DV, alg.alg)
    ind = findtruncated(diagview(D), alg.trunc)
    return truncate!((D, V), ind)
end

copy_input(::typeof(eig_trunc), A) = copy_input(eig_full, A)

function select_algorithm(::typeof(eig_trunc!), A::AbstractMatrix; kwargs...)
    return TruncatedAlgorithm(default_eig_algorithm(A; kwargs...), NoTruncation())
end
function initialize_output(::typeof(eig_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm)
    return initialize_output(eig_full!, A, alg.alg)
end
