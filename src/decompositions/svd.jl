# SVD API
# -------
# TODO: do not export but mark as public ?
function svd!(A::AbstractMatrix, args...; kwargs...)
    return svd_compact!(A, args...; kwargs...)
end

"""
    svd_full(A; kwargs...) -> U, S, Vᴴ
    svd_full(A, alg::AbstractAlgorithm) -> U, S, Vᴴ
    svd_full!(A, [USVᴴ]; kwargs...) -> U, S, Vᴴ
    svd_full!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ

Compute the singular value decomposition (SVD) of `A`, such that `A = U * S * Vᴴ`.
The full version produces components such that for an M×N matrix `A`,
both `U` and `Vᴴ` are square and unitary, of size M×M and N×N respectively.

See also [`svd_compact(!)`](@ref svd_compact), [`svd_vals(!)`](@ref svd_vals),
[`svd_trunc(!)`](@ref svd_trunc) and [`svd_null(!)`](@ref svd_null).
"""
@functiondef svd_full

"""
    svd_compact(A; kwargs...) -> U, S, Vᴴ
    svd_compact(A, alg::AbstractAlgorithm) -> U, S, Vᴴ
    svd_compact!(A, [USVᴴ]; kwargs...) -> U, S, Vᴴ
    svd_compact!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ

Compute the singular value decomposition (SVD) of `A`, such that `A = U * S * Vᴴ`.
The compact version produces components such that for an M×N matrix `A`,
`S` is square of size K×K with `K = min(M, N)`, and `U` and `Vᴴ` are isometries
of size M×K and K×N respectively.

See also [`svd_full(!)`](@ref svd_full), [`svd_vals(!)`](@ref svd_vals),
[`svd_trunc(!)`](@ref svd_trunc) and [`svd_null(!)`](@ref svd_null).
"""
@functiondef svd_compact

# TODO: decide if we should have `svd_trunc!!` instead
"""
    svd_trunc(A; kwargs...) -> U, S, Vᴴ
    svd_trunc(A, alg::AbstractAlgorithm) -> U, S, Vᴴ
    svd_trunc!(A, [USVᴴ]; kwargs...) -> U, S, Vᴴ
    svd_trunc!(A, [USVᴴ], alg::AbstractAlgorithm) -> U, S, Vᴴ

Compute the truncated singular value decomposition (SVD) of `A`, such that `A ≈ U * S * Vᴴ`.
The truncated version produces components such that for an M×N matrix `A`,
`S` is square of size K×K with K the number of kept singular values,
and `U` and `Vᴴ` are isometries of size M×K and K×N respectively.

Depending on the `alg`, the input `USVᴴ` can sometimes be either ignored or its
memory recycled, such that the actual output does not need to, but can coincide
with the provided `USVᴴ`.

See also [`svd_full(!)`](@ref svd_full), [`svd_compact(!)`](@ref svd_compact),
[`svd_vals(!)`](@ref svd_vals) and [`svd_null(!)`](@ref svd_null).
"""
@functiondef svd_trunc

# TODO: could be `nullspace` with a `SVDAlgorithm` instead. 
# TODO: update docs for kwargs?
"""
    svd_null(A; kwargs...) -> N
    svd_null(A, alg::AbstractAlgorithm) -> N
    svd_null!(A, [USVᴴ]; kwargs...) -> N
    svd_null!(A, [USVᴴ], alg::AbstractAlgorithm) -> N

Compute a basis for the nullspace of `A`, such that `A * N ≈ 0`. This is done
by including the singular vectors of `A` whose singular values have magnitudes
smaller than `max(atol, rtol*σ₁)`, where `σ₁` is the largest singular value.

See also [`svd_full(!)`](@ref svd_full), [`svd_compact(!)`](@ref svd_compact),
[`svd_trunc(!)`](@ref svd_trunc) and [`svd_vals(!)`](@ref svd_vals).
"""
@functiondef svd_null

"""
    svd_vals(A; kwargs...) -> S
    svd_vals(A, alg::AbstractAlgorithm) -> S
    svd_vals!(A, [S]; kwargs...) -> S
    svd_vals!(A, [S], alg::AbstractAlgorithm) -> S

Compute the vector of singular values of `A`, such that for an M×N matrix `A`,
`S` is a vector of size `K = min(M, N)`, the number of kept singular values.

See also [`svd_full(!)`](@ref svd_full), [`svd_compact(!)`](@ref svd_compact),
[`svd_trunc(!)`](@ref svd_trunc) and [`svd_null(!)`](@ref svd_null).
"""
@functiondef svd_vals

# Default to LAPACK sdd for `StridedMatrix{<:BlasFloat}`
function default_svd_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_DivideAndConquer(; kwargs...)
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
function initialize_output(::typeof(svd_full!), A::AbstractMatrix, ::LAPACK_SVDAlgorithm)
    m, n = size(A)
    U = similar(A, (m, m))
    S = similar(A, real(eltype(A)), (m, n))
    Vᴴ = similar(A, (n, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_compact!), A::AbstractMatrix, ::LAPACK_SVDAlgorithm)
    m, n = size(A)
    minmn = min(m, n)
    U = similar(A, (m, minmn))
    S = Diagonal(similar(A, real(eltype(A)), (minmn,)))
    Vᴴ = similar(A, (minmn, n))
    return (U, S, Vᴴ)
end
function initialize_output(::typeof(svd_vals!), A::AbstractMatrix, ::LAPACK_SVDAlgorithm)
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

# check input
function check_input(::typeof(svd_full!), A::AbstractMatrix, USVᴴ)
    m, n = size(A)
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
    _, _, Vᴴ = svd_full!(A, alg)
    i = findfirst(<=(atol), diag(S))
    if isnothing(i)
        i = min(m, n) + 1
    end
    return Vᴴ[i:end, :]'
end

function svd_trunc!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm)
    U, S, Vᴴ = svd_compact!(A, USVᴴ, alg.alg)
    ind = findtruncated(diagview(S), alg.trunc)
    return truncate!((U, S, Vᴴ), ind)
end

copy_input(::typeof(svd_trunc), A) = copy_input(svd_compact, A)
function select_algorithm(::typeof(svd_trunc!), A; kwargs...)
    return TruncatedAlgorithm(default_svd_algorithm(A; kwargs...), NoTruncation())
end
function initialize_output(::typeof(svd_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm)
    return initialize_output(svd_compact!, A, alg.alg)
end
