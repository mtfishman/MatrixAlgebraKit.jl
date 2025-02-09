# TODO: figure out what the "output" should be:
# - USVᴴ for usage with the full svd?
# - N for usage with nullspace?
# - USVᴴN for both?

# for now: pretend this is just svd_trunc with svd_full as a driver

# Inputs
# ------
copy_input(::typeof(left_null), A::AbstractMatrix) = copy_input(svd_full, A)
copy_input(::typeof(right_null), A::AbstractMatrix) = copy_input(svd_full, A)

check_input(::typeof(left_null), A::AbstractMatrix, USVᴴ) = check_input(svd_full!, A, USVᴴ)
check_input(::typeof(right_null), A::AbstractMatrix, USVᴴ) = check_input(svd_full!, A, USVᴴ)

# Outputs
# -------
function initialize_output(::typeof(left_null!), A::AbstractMatrix, alg::TruncatedAlgorithm)
    return initialize_output(svd_full!, A, alg.alg)
end
function initialize_output(::typeof(right_null!), A::AbstractMatrix,
                           alg::TruncatedAlgorithm)
    return initialize_output(svd_full!, A, alg.alg)
end

# Implementation
# --------------
function left_null!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm)
    U, S, _ = svd_full!(A, USVᴴ, alg.alg)
    atol = max(alg.trunc.atol, alg.trunc.rtol * first(S))
    i = @something findfirst(≤(atol), diagview(S)) minimum(size(A)) + 1
    return U[:, i:end]'
end
function right_null!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm)
    _, S, Vᴴ = svd_full!(A, USVᴴ, alg.alg)
    atol = max(alg.trunc.atol, alg.trunc.rtol * first(S))
    i = @something findfirst(≤(atol), diagview(S)) minimum(size(A)) + 1
    return Vᴴ[i:end, :]'
end
