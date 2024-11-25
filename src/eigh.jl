# TODO: do not export but mark as public ?
function eigh!(A::AbstractMatrix, args...; kwargs...)
    return eigh_full!(A, args...; kwargs...)
end

function eigh_full!(A::AbstractMatrix,
                    D::AbstractVector=similar(A, real(eltype(A)), size(A, 1)),
                    V::AbstractMatrix=similar(A, size(A));
                    kwargs...)
    return eigh_full!(A, D, V, default_backend(eigh_full!, A; kwargs...); kwargs...)
end
function eigh_vals!(A::AbstractMatrix,
                    D::AbstractVector=similar(A, real(eltype(A)), size(A, 1));
                    kwargs...)
    return eigh_vals!(A, D, default_backend(eigh_vals!, A; kwargs...); kwargs...)
end
function eigh_trunc!(A::AbstractMatrix;
                     kwargs...)
    return eigh_trunc!(A, default_backend(eigh_trunc!, A; kwargs...); kwargs...)
end

function default_backend(::typeof(eigh_full!), A::AbstractMatrix; kwargs...)
    return default_eigh_backend(A; kwargs...)
end
function default_backend(::typeof(eigh_vals!), A::AbstractMatrix; kwargs...)
    return default_eigh_backend(A; kwargs...)
end
function default_backend(::typeof(eigh_trunc!), A::AbstractMatrix; kwargs...)
    return default_eigh_backend(A; kwargs...)
end

function default_eigh_backend(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACKBackend()
end

function check_eigh_full_input(A, D, V)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square matrix"))
    size(D) == (n,) ||
        throw(DimensionMismatch("Eigenvalue vector `D` must have length equal to size(A, 1)"))
    size(V) == (n, n) ||
        throw(DimensionMismatch("Eigenvector matrix `V` must have size equal to A"))
    return nothing
end
function check_eigh_vals_input(A, D)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square matrix"))
    size(D) == (n,) ||
        throw(DimensionMismatch("Eigenvalue vector `D` must have length equal to size(A, 1)"))
    return nothing
end

@static if VERSION >= v"1.12-DEV.0"
    const RobustRepresentations = LinearAlgebra.RobustRepresentations
else
    struct RobustRepresentations end
end

function eigh_full!(A::AbstractMatrix,
                    D::AbstractVector,
                    V::AbstractMatrix,
                    backend::LAPACKBackend;
                    alg=RobustRepresentations(),
                    kwargs...)
    check_eigh_full_input(A, D, V)
    if alg == RobustRepresentations()
        YALAPACK.heevr!(A, D, V; kwargs...)
    elseif alg == LinearAlgebra.DivideAndConquer()
        YALAPACK.heevd!(A, D, V; kwargs...)
    elseif alg == LinearAlgebra.QRIteration()
        YALAPACK.heev!(A, D, V; kwargs...)
    else
        throw(ArgumentError("Unknown LAPACK eigenvalue algorithm $alg"))
    end
    return D, V
end

function eigh_vals!(A::AbstractMatrix,
                    D::AbstractVector,
                    backend::LAPACKBackend;
                    alg=RobustRepresentations(),
                    kwargs...)
    check_eigh_vals_input(A, D)
    V = similar(A, (size(A, 1), 0))
    if alg == RobustRepresentations()
        YALAPACK.heevr!(A, D, V; kwargs...)
    elseif alg == LinearAlgebra.DivideAndConquer()
        YALAPACK.heevd!(A, D, V; kwargs...)
    elseif alg == LinearAlgebra.QRIteration()
        YALAPACK.heev!(A, D, V; kwargs...)
    else
        throw(ArgumentError("Unknown LAPACK eigenvalue algorithm $alg"))
    end
    return D
end

# for eigh_trunc!, it doesn't make sense to preallocate D and V as we don't know their sizes
function eigh_trunc!(A::AbstractMatrix,
                     backend::LAPACKBackend;
                     alg=RobustRepresentations(),
                     atol=zero(real(eltype(A))),
                     rtol=zero(real(eltype(A))),
                     rank=size(A, 1),
                     kwargs...)
    if alg == RobustRepresentations()
        D, V = YALAPACK.heevr!(A; kwargs...)
    elseif alg == LinearAlgebra.DivideAndConquer()
        D, V = YALAPACK.heevd!(A; kwargs...)
    elseif alg == LinearAlgebra.QRIteration()
        D, V = YALAPACK.heev!(A; kwargs...)
    else
        throw(ArgumentError("Unknown LAPACK eigenvalue algorithm $alg"))
    end
    # eigenvalues are sorted in ascending order
    # TODO: do we assume that they are positive, or should we check for this?
    # or do we want to truncate based on absolute value and thus sort differently?
    n = length(D)
    tol = convert(eltype(D), max(atol, rtol * D[n]))
    s = max(n - rank + 1, findfirst(>=(tol), D))
    # TODO: do we want views here, such that we do not need extra allocations if we later
    # copy them into other storage
    return D[n:-1:s], V[:, n:-1:s]
end
