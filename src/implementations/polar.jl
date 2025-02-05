# Inputs
# ------
copy_input(::typeof(left_polar), A::AbstractMatrix) = copy_input(svd_full, A)
copy_input(::typeof(right_polar), A::AbstractMatrix) = copy_input(svd_full, A)

function check_input(::typeof(left_polar!), A::AbstractMatrix, WP)
    m, n = size(A)
    W, P = WP
    m >= n ||
        throw(ArgumentError("`left_polar!` requires a matrix A with at least as many rows as columns"))
    (W isa AbstractMatrix && eltype(W) == eltype(A) && size(W) == (m, n)) ||
        throw(ArgumentError("`left_polar!` requires a matrix W with the same size and eltype as A"))
    (P isa AbstractMatrix && eltype(P) == eltype(A) && size(P) == (n, n)) ||
        throw(ArgumentError("`left_polar!` requires a square matrix P with the same eltype and number of columns as A"))
    return nothing
end
function check_input(::typeof(right_polar!), A::AbstractMatrix, PWᴴ)
    m, n = size(A)
    P, Wᴴ = PWᴴ
    n >= m ||
        throw(ArgumentError("`right_polar!` requires a matrix A with at least as many columns as rows"))
    (P isa AbstractMatrix && eltype(P) == eltype(A) && size(P) == (m, m)) ||
        throw(ArgumentError("`right_polar!` requires a square matrix P with the same eltype and number of rows as A"))
    (Wᴴ isa AbstractMatrix && eltype(Wᴴ) == eltype(A) && size(Wᴴ) == (m, n)) ||
        throw(ArgumentError("`right_polar!` requires a matrix Wᴴ with the same size and eltype as A"))
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(left_polar!), A::AbstractMatrix, ::LAPACK_SVDAlgorithm)
    m, n = size(A)
    W = similar(A)
    P = similar(A, (n, n))
    return (W, P)
end
function initialize_output(::typeof(right_polar!), A::AbstractMatrix, ::LAPACK_SVDAlgorithm)
    m, n = size(A)
    P = similar(A, (m, m))
    Wᴴ = similar(A)
    return (P, Wᴴ)
end

# Implementation
# --------------
function left_polar!(A::AbstractMatrix, WP, alg::LAPACK_SVDAlgorithm)
    check_input(left_polar!, A, WP)
    U, S, Vᴴ = svd_compact!(A, alg)
    W, P = WP
    W = mul!(W, U, Vᴴ)
    S .= sqrt.(S)
    SsqrtVᴴ = lmul!(S, Vᴴ)
    P = mul!(P, SsqrtVᴴ', SsqrtVᴴ)
    return (W, P)
end
function right_polar!(A::AbstractMatrix, PWᴴ, alg::LAPACK_SVDAlgorithm)
    check_input(right_polar!, A, PWᴴ)
    U, S, Vᴴ = svd_compact!(A, alg)
    P, Wᴴ = PWᴴ
    Wᴴ = mul!(Wᴴ, U, Vᴴ)
    S .= sqrt.(S)
    USsqrt = rmul!(U, S)
    P = mul!(P, USsqrt, USsqrt')
    return (P, Wᴴ)
end
