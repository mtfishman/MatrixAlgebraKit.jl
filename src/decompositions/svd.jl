# TODO: do not export but mark as public ?
function svd!(A::AbstractMatrix, args...; kwargs...)
    return svd_compact!(A, args...; kwargs...)
end

# copy input
function copy_input(::typeof(svd_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(svd_compact), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(svd_null), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(svd_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

# initialize output
function initialize_output(::typeof(svd_full!), A::AbstractMatrix)
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, m))
    S = similar(A, real(eltype(A)), (m, n))
    Vᴴ = similar(A, (n, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_compact!), A::AbstractMatrix)
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, minmn))
    S = Diagonal(similar(A, real(eltype(A)), (minmn,)))
    Vᴴ = similar(A, (minmn, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_vals!), A::AbstractMatrix)
    return similar(A, real(eltype(A)), (min(size(A)...),))
end

# select default algorithm
function select_algorithm(::typeof(svd_full!), A; kwargs...)
    return default_svd_algorithm(A; kwargs...)
end
function select_algorithm(::typeof(svd_compact!), A; kwargs...)
    return default_svd_algorithm(A; kwargs...)
end
function select_algorithm(::typeof(svd_vals!), A; kwargs...)
    return default_svd_algorithm(A; kwargs...)
end
function select_algorithm(::typeof(svd_null!), A; kwargs...)
    return default_svd_algorithm(A; kwargs...)
end

function default_svd_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_DivideAndConquer(; kwargs...)
end

# check input
function check_input(::typeof(svd_full!), A::AbstractMatrix, USVᴴ)
    m, n = size(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    (U isa AbstractMatrix && eltype(U) == eltype(A) && size(U) == (m, m)) ||
        throw(ArgumentError("`svd_full!` requires square U matrix with equal number of rows and same `eltype` as A"))
    (Vᴴ isa AbstractMatrix && eltype(Vᴴ) == eltype(A) && size(Vᴴ) == (n, n)) ||
        throw(ArgumentError("`svd_full!` requires square Vᴴ matrix with equal number of columns and same `eltype` as A"))
    (S isa AbstractMatrix && eltype(S) == real(eltype(A)) && size(S) == (m, n)) ||
        throw(ArgumentError("`svd_full!` requires a matrix S of the same size as A with a real `eltype`"))
    return nothing
end
function check_input(::typeof(svd_compact!), A::AbstractMatrix, USVᴴ)
    m, n = size(A)
    minmn = min(m, n)
    U, S, Vᴴ = USVᴴ
    (U isa AbstractMatrix && eltype(U) == eltype(A) && size(U) == (m, minmn)) ||
        throw(ArgumentError("`svd_full!` requires square U matrix with equal number of rows and same `eltype` as A"))
    (Vᴴ isa AbstractMatrix && eltype(Vᴴ) == eltype(A) && size(Vᴴ) == (minmn, n)) ||
        throw(ArgumentError("`svd_full!` requires square Vᴴ matrix with equal number of columns and same `eltype` as A"))
    (S isa Diagonal && eltype(S) == real(eltype(A)) && size(S) == (minmn, minmn)) ||
        throw(ArgumentError("`svd_compact!` requires Diagonal matrix S with number of rows equal to min(size(A)...) with a real `eltype`"))
    return nothing
end
function check_input(::typeof(svd_vals!), A::AbstractMatrix, S)
    m, n = size(A)
    minmn = min(m, n)
    (S isa AbstractVector && eltype(S) == real(eltype(A)) && size(S) == (minmn,)) ||
        throw(ArgumentError("`svd_vals!` requires vector S of length min(size(A)...) with a real `eltype`"))
    return nothing
end

# actual implementation
function svd_full!(A::AbstractMatrix, USVᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(svd_full!, A, USVᴴ)
    U, S, Vᴴ = USVᴴ
    fill!(S, zero(eltype(S)))
    minmn = min(size(A)...)
    if alg isa LAPACK_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_QRIteration does not accept any keyword arguments"))
        YALAPACK.gesvd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        YALAPACK.gesdd!(A, view(S, 1:minmn, 1), U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        throw(ArgumentError("LAPACK_Bisection is not supported for full SVD"))
    elseif alg isa LAPACK_Jacobi
        throw(ArgumentError("LAPACK_Bisection is not supported for full SVD"))
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    for i in 2:minmn
        S[i, i] = S[i, 1]
        S[i, 1] = zero(eltype(S))
    end
    return USVᴴ
end
function svd_compact!(A::AbstractMatrix, USVᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(svd_compact!, A, USVᴴ)
    U, S, Vᴴ = USVᴴ
    if alg isa LAPACK_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_QRIteration does not accept any keyword arguments"))
        YALAPACK.gesvd!(A, S.diag, U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        YALAPACK.gesdd!(A, S.diag, U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        YALAPACK.gesvdx!(A, S.diag, U, Vᴴ; alg.kwargs...)
    elseif alg isa LAPACK_Jacobi
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Jacobi does not accept any keyword arguments"))
        YALAPACK.gesvj!(A, S.diag, U, Vᴴ)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    return USVᴴ
end
function svd_vals!(A::AbstractMatrix, S, alg::LAPACK_SVDAlgorithm)
    check_input(svd_vals!, A, S)
    U, Vᴴ = similar(A, (0, 0)), similar(A, (0, 0))
    if alg isa LAPACK_QRIteration
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_QRIteration does not accept any keyword arguments"))
        YALAPACK.gesvd!(A, S, U, Vᴴ)
    elseif alg isa LAPACK_DivideAndConquer
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_DivideAndConquer does not accept any keyword arguments"))
        YALAPACK.gesdd!(A, S, U, Vᴴ)
    elseif alg isa LAPACK_Bisection
        YALAPACK.gesvdx!(A, S, U, Vᴴ; alg.kwargs...)
    elseif alg isa LAPACK_Jacobi
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Jacobi does not accept any keyword arguments"))
        YALAPACK.gesvj!(A, S, U, Vᴴ)
    else
        throw(ArgumentError("Unsupported SVD algorithm"))
    end
    return S
end
function svd_null!(A::AbstractMatrix, alg::LAPACK_SVDAlgorithm; atol)
    m, n = size(A)
    USVᴴ = svd_full!(A, alg)
    i = findfirst(<=(atol), diag(S))
    if isnothing(i)
        i = min(m, n) + 1
    end
    return Vᴴ[i:end, :]'
end

# for svd_trunc!, it doesn't make sense to preallocate U, S, Vᴴ as we don't know their sizes
# function select_algorithm(::typeof(svd_trunc!), A; kwargs...)
#     return default_svd_algorithm(A; kwargs...)
# end

# function svd_trunc!(A::AbstractMatrix, alg::LAPACK_SVDAlgorithm, trunc::TruncationStrategy)
#     U, S, Vᴴ = svd_compact(A, alg)

#     Sd = S.diag
#     ind = findtruncated(Sd, trunc)
#     U′ = U[:, ind]
#     S′ = Diagonal(Sd[ind])
#     Vᴴ′ = Vᴴ[ind, :]
#     return (U′, S′, Vᴴ′)
# end
