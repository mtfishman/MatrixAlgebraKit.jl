# Schur functions
# -------------

"""
    schur_full(A; kwargs...) -> T, Z, vals
    schur_full(A, alg::AbstractAlgorithm) -> T, Z, vals
    schur_full!(A, [TZv]; kwargs...) -> T, Z, vals
    schur_full!(A, [TZv], alg::AbstractAlgorithm) -> T, Z, vals

Compute the full Schur decomposition of the square matrix `A`,
such that `A * Z = Z * T`, where the orthogonal or unitary matrix `Z` contains the
Schur vectors and the square matrix `T` is upper triangular (in the complex case)
or quasi-upper triangular (in the real case). The list `vals` contains the (complex-valued)
eigenvalues of `A`, as extracted from the (quasi-)diagonal of `T`.

!!! note
    The bang method `schur_full!` optionally accepts the output structure and
    possibly destroys the input matrix `A`. Always use the return value of the function
    as it may not always be possible to use the provided `TZv` as output.
"""
@functiondef schur_full

# TODO: partial or truncated schur? Do we ever want or use this?

# Algorithm selection
# -------------------
function select_algorithm(::typeof(schur_full), A; kwargs...)
    return select_algorithm(schur_full!, A; kwargs...)
end
function select_algorithm(::typeof(schur_full!), A; alg=nothing, kwargs...)
    if alg isa AbstractAlgorithm
        return alg
    elseif alg isa Symbol
        return Algorithm{alg}(; kwargs...)
    else
        isnothing(alg) || throw(ArgumentError("Unknown alg $alg"))
        return default_eig_algorithm(A; kwargs...)
    end
end
