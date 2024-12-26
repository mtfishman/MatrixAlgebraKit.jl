# TODO: export? or not export but mark as public ?
function eigh!(A::AbstractMatrix, args...; kwargs...)
    return eigh_full!(A, args...; kwargs...)
end

function eigh_full!(A::AbstractMatrix, DV=eigh_full_init(A); kwargs...)
    return eigh_full!(A, DV, default_algorithm(eigh_full!, A; kwargs...))
end
function eigh_vals!(A::AbstractMatrix, D=eigh_vals_init(A); kwargs...)
    return eigh_vals!(A, D, default_algorithm(eigh_vals!, A; kwargs...))
end
function eigh_trunc!(A::AbstractMatrix; kwargs...)
    return eigh_trunc!(A, default_algorithm(eigh_trunc!, A; kwargs...))
end

function eigh_full_init(A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    D = similar(A, real(eltype(A)), n)
    V = similar(A, (n, n))
    return (D, V)
end
function eigh_vals_init(A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    D = similar(A, real(eltype(A)), n)
    return D
end

function default_algorithm(::typeof(eigh_full!), A::AbstractMatrix; kwargs...)
    return default_eigh_algorithm(A; kwargs...)
end
function default_algorithm(::typeof(eigh_vals!), A::AbstractMatrix; kwargs...)
    return default_eigh_algorithm(A; kwargs...)
end
function default_algorithm(::typeof(eigh_trunc!), A::AbstractMatrix; kwargs...)
    return default_eigh_algorithm(A; kwargs...)
end

function default_eigh_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_RobustRepresentations(; kwargs...)
end

function check_eigh_full_input(A::AbstractMatrix, (D, V))
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square matrix"))
    size(D) == (n,) ||
        throw(DimensionMismatch("Eigenvalue vector `D` must have length equal to size(A, 1)"))
    size(V) == (n, n) ||
        throw(DimensionMismatch("Eigenvector matrix `V` must have size equal to A"))
    return nothing
end
function check_eigh_vals_input(A::AbstractMatrix, (D, V))
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square matrix"))
    size(D) == (n,) ||
        throw(DimensionMismatch("Eigenvalue vector `D` must have length equal to size(A, 1)"))
    return nothing
end

const LAPACK_EighAlgorithm = Union{LAPACK_RobustRepresentations,LAPACK_QRIteration,
                                   LAPACK_DivideAndConquer}
function eigh_full!(A::AbstractMatrix, DV, alg::LAPACK_EighAlgorithm)
    check_eigh_full_input(A, DV)
    D, V = DV
    if alg isa LAPACK_RobustRepresentations
        YALAPACK.heevr!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, D, V; alg.kwargs...)
    else
        YALAPACK.heev!(A, D, V; alg.kwargs...)
    end
    return D, V
end

function eigh_vals!(A::AbstractMatrix, D, alg::LAPACK_EighAlgorithm)
    check_eigh_vals_input(A, D)
    V = similar(A, (size(A, 1), 0))
    if alg isa LAPACK_RobustRepresentations
        YALAPACK.heevr!(A, D, V; alg.kwargs...)
    elseif alg isa LAPACK_DivideAndConquer
        YALAPACK.heevd!(A, D, V; alg.kwargs...)
    else
        YALAPACK.heev!(A, D, V; alg.kwargs...)
    end
    return D, V
end

# for eigh_trunc!, it doesn't make sense to preallocate D and V as we don't know their sizes
# function eigh_trunc!(A::AbstractMatrix,
#                      backend::LAPACKBackend;
#                      alg=RobustRepresentations(),
#                      atol=zero(real(eltype(A))),
#                      rtol=zero(real(eltype(A))),
#                      rank=size(A, 1),
#                      kwargs...)
#     if alg == RobustRepresentations()
#         D, V = YALAPACK.heevr!(A; kwargs...)
#     elseif alg == LinearAlgebra.DivideAndConquer()
#         D, V = YALAPACK.heevd!(A; kwargs...)
#     elseif alg == LinearAlgebra.QRIteration()
#         D, V = YALAPACK.heev!(A; kwargs...)
#     else
#         throw(ArgumentError("Unknown LAPACK eigenvalue algorithm $alg"))
#     end
#     # eigenvalues are sorted in ascending order
#     # TODO: do we assume that they are positive, or should we check for this?
#     # or do we want to truncate based on absolute value and thus sort differently?
#     n = length(D)
#     tol = convert(eltype(D), max(atol, rtol * D[n]))
#     s = max(n - rank + 1, findfirst(>=(tol), D))
#     # TODO: do we want views here, such that we do not need extra allocations if we later
#     # copy them into other storage
#     return D[n:-1:s], V[:, n:-1:s]
# end
