# Inputs
# ------
function copy_input(::typeof(svd_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(svd_compact), A::AbstractMatrix) = copy_input(svd_full, A)
copy_input(::typeof(svd_vals), A::AbstractMatrix) = copy_input(svd_full, A)
# copy_input(::typeof(svd_null), A::AbstractMatrix) = copy_input(svd_full, A)
copy_input(::typeof(svd_trunc), A) = copy_input(svd_compact, A)

# TODO: many of these checks are happening again in the LAPACK routines
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

# Outputs
# -------
function initialize_output(::typeof(svd_full!), A::AbstractMatrix, ::LAPACK_SVDAlgorithm)
    m, n = size(A)
    U = similar(A, (m, m))
    S = similar(A, real(eltype(A)), (m, n)) # TODO: Rectangular diagonal type?
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
function initialize_output(::typeof(svd_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm)
    return initialize_output(svd_compact!, A, alg.alg)
end

# Implementation
# --------------
function svd_full!(A::AbstractMatrix, USVᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(svd_full!, A, USVᴴ)
    U, S, Vᴴ = USVᴴ
    fill!(S, zero(eltype(S)))
    m, n = size(A)
    minmn = min(m, n)
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
    # TODO: make this controllable using a `gaugefix` keyword argument
    for j in 1:max(m, n)
        if j <= minmn
            u = view(U, :, j)
            v = view(Vᴴ, j, :)
            s = conj(sign(argmax(abs, u)))
            u .*= s
            v .*= conj(s)
        elseif j <= m
            u = view(U, :, j)
            s = conj(sign(argmax(abs, u)))
            u .*= s
        else
            v = view(Vᴴ, j, :)
            s = conj(sign(argmax(abs, v)))
            v .*= s
        end
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
    # TODO: make this controllable using a `gaugefix` keyword argument
    for j in 1:size(U, 2)
        u = view(U, :, j)
        v = view(Vᴴ, j, :)
        s = conj(sign(argmax(abs, u)))
        u .*= s
        v .*= conj(s)
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
# function svd_null!(A::AbstractMatrix, alg::LAPACK_SVDAlgorithm)
#     m, n = size(A)
#     _, _, Vᴴ = svd_full!(A, alg)
#     atol = alg.atol
#     i = findfirst(<=(atol), diag(S))
#     if isnothing(i)
#         i = min(m, n) + 1
#     end
#     return Vᴴ[i:end, :]'
# end

function svd_trunc!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm)
    U, S, Vᴴ = svd_compact!(A, USVᴴ, alg.alg)
    ind = findtruncated(diagview(S), alg.trunc)
    return truncate!((U, S, Vᴴ), ind)
end
