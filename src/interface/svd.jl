# SVD functions
# -------------
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

# Algorithm selection
# -------------------
for f in (:svd_full, :svd_compact, :svd_vals, :svd_null)
    f! = Symbol(f, :!)
    @eval begin
        function select_algorithm(::typeof($f), A; kwargs...)
            return select_algorithm($f!, A; kwargs...)
        end
        function select_algorithm(::typeof($f!), A; alg=nothing, kwargs...)
            if alg isa AbstractAlgorithm
                return alg
            elseif alg isa Symbol
                return Algorithm{alg}(; kwargs...)
            else
                isnothing(alg) || throw(ArgumentError("Unknown alg $alg"))
                return default_svd_algorithm(A; kwargs...)
            end
        end
    end
end

function select_algorithm(::typeof(svd_trunc), A; kwargs...)
    return select_algorithm(svd_trunc!, A; kwargs...)
end
function select_algorithm(::typeof(svd_trunc!), A; alg=nothing, trunc=nothing, kwargs...)
    alg_svd = select_algorithm(svd_compact!, A; alg, kwargs...)
    alg_trunc = trunc isa TruncationStrategy ? trunc :
                trunc isa NamedTuple ? TruncationStrategy(; trunc...) :
                isnothing(trunc) ? NoTruncation() :
                throw(ArgumentError("Unknown truncation strategy: $trunc"))
    return TruncatedAlgorithm(alg_svd, alg_trunc)
end

# Default to LAPACK SDD for `StridedMatrix{<:BlasFloat}`
function default_svd_algorithm(A::StridedMatrix{<:BlasFloat}; kwargs...)
    isempty(kwargs) || throw(ArgumentError("Unexpected kwargs: $kwargs"))
    return LAPACK_DivideAndConquer(; kwargs...)
end
