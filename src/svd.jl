function svd_full!(A::AbstractMatrix,
                   U::AbstractMatrix=similar(A, (size(A, 1), size(A, 1))),
                   S::AbstractVector=similar(A, real(eltype(A)), (min(size(A)...),)),
                   Vᴴ::AbstractMatrix=similar(A, (size(A, 2), size(A, 2)));
                   kwargs...)
    return svd_full!(A, U, S, Vᴴ, default_backend(svd_full!, A; kwargs...); kwargs...)
end
function svd_compact!(A::AbstractMatrix,
                      U::AbstractMatrix=similar(A, (size(A, 1), size(A, 1))),
                      S::AbstractVector=similar(A, real(eltype(A)), (min(size(A)...),)),
                      Vᴴ::AbstractMatrix=similar(A, (size(A, 2), size(A, 2)));
                      kwargs...)
    return svd_compact!(A, U, S, Vᴴ, default_backend(svd_compact!, A; kwargs...); kwargs...)
end
function svd_trunc!(A::AbstractMatrix;
                    kwargs...)
    return svd_trunc!(A, default_backend(svd_trunc!, A; kwargs...); kwargs...)
end

function default_backend(::typeof(svd_full!), A::AbstractMatrix; kwargs...)
    return default_svd_backend(A; kwargs...)
end
function default_backend(::typeof(svd_compact!), A::AbstractMatrix; kwargs...)
    return default_svd_backend(A; kwargs...)
end
function default_backend(::typeof(svd_trunc!), A::AbstractMatrix; kwargs...)
    return default_svd_backend(A; kwargs...)
end

function default_svd_backend(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACKBackend()
end

function check_svd_full_input(A, U, S, Vᴴ)
    m, n = size(A)
    minmn = min(m, n)
    size(U) == (m, m) ||
        throw(DimensionMismatch("`svd_full!` requires square U matrix with equal number of rows as A"))
    size(Vᴴ) == (n, n) ||
        throw(DimensionMismatch("`svd_full!` requires square Vᴴ matrix with equal number of columns as A"))
    size(S) == (minmn,) ||
        throw(DimensionMismatch("`svd_full!` requires vector S of length min(size(A)..."))
    return nothing
end
function check_svd_compact_input(A, U, S, Vᴴ)
    m, n = size(A)
    minmn = min(m, n)
    size(U) == (m, minmn) ||
        throw(DimensionMismatch("`svd_compact!` requires square U matrix with equal number of rows as A"))
    size(Vᴴ) == (minmn, n) ||
        throw(DimensionMismatch("`svd_compact!` requires square Vᴴ matrix with equal number of columns as A"))
    size(S) == (minmn,) ||
        throw(DimensionMismatch("`svd_compact!` requires vector S of length min(size(A)..."))
    return nothing
end

function svd_full!(A::AbstractMatrix,
                   U::AbstractMatrix,
                   S::AbstractVector,
                   Vᴴ::AbstractMatrix,
                   backend::LAPACKBackend;
                   alg=LinearAlgebra.DivideAndConquer())
    check_svd_full_input(A, U, S, Vᴴ)
    if alg == LinearAlgebra.DivideAndConquer()
        YALAPACK.gesdd!(A, S, U, Vᴴ)
    elseif alg == LinearAlgebra.QRIteration()
        YALAPACK.gesvd!(A, S, U, Vᴴ)
    else
        throw(ArgumentError("Unknown algorithm $alg"))
    end
    return U, S, Vᴴ
end
function svd_compact!(A::AbstractMatrix,
                      U::AbstractMatrix,
                      S::AbstractVector,
                      Vᴴ::AbstractMatrix,
                      backend::LAPACKBackend;
                      alg=LinearAlgebra.DivideAndConquer())
    check_svd_compact_input(A, U, S, Vᴴ)
    if alg == LinearAlgebra.DivideAndConquer()
        YALAPACK.gesdd!(A, S, U, Vᴴ)
    elseif alg == LinearAlgebra.QRIteration()
        YALAPACK.gesvd!(A, S, U, Vᴴ)
    else
        throw(ArgumentError("Unknown algorithm $alg"))
    end
    return U, S, Vᴴ
end

# for svd_trunc!, it doesn't make sense to preallocate U, S, Vᴴ as we don't know their sizes
function svd_trunc!(A::AbstractMatrix,
                    backend::LAPACKBackend;
                    alg=LinearAlgebra.DivideAndConquer(),
                    tol=zero(real(eltype(A))),
                    rank=min(size(A)...))
    if alg == LinearAlgebra.DivideAndConquer()
        S, U, Vᴴ = YALAPACK.gesdd!(A)
    elseif alg == LinearAlgebra.QRIteration()
        S, U, Vᴴ = YALAPACK.gesvd!(A)
    else
        throw(ArgumentError("Unknown algorithm $alg"))
    end
    r = min(rank, findlast(>=(tol * S[1]), S))
    # TODO: do we want views here, such that we do not need extra allocations if we later
    # copy them into other storage
    return U[:, 1:r], S[1:r], Vᴴ[1:r, :]
end