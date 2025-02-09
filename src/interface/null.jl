# TODO: do we want `A * N'` or `A * N`?

# Null API
# ---------
function null!(A::AbstractMatrix, args...; kwargs...)
    return left_null!(A, args...; kwargs...)
end
function null(A::AbstractMatrix, args...; kwargs...)
    return left_null(A, args...; kwargs...)
end

# Null functions
# --------------
"""
    left_null(A; atol::Real, rtol::Real) -> N
    left_null(A, alg::AbstractAlgorithm) -> N
    left_null!(A, [N]; atol::Real, rtol::Real) -> N
    left_null!(A, [N], alg::AbstractAlgorithm) -> N

Compute the left nullspace of the matrix `A` of size `(m, n)` with `m ≥ n`,
such that `N * A ≈ 0`. Here, `N` is a matrix with orthonormal rows that span the
cokernel of `A`. By default, this is achieved by including the left singular vectors
whose singular values have magnitude smaller than `max(atol, rtol * σ₁)`, where `σ₁`
is `A`'s largest singular value.

The keyword arguments can be used to control the precision and have default values
`atol=0` and `rtol=atol > 0 ? 0 : n*ϵ`, where `n` is the size of the smallest dimension of `A`,
and `ϵ` is the `eps` of the element type.

!!! note
    The bang method `left_null!` optionally accepts the output structure and possibly destroys
    the input matrix `A`. Always use the return value of the function as it may not always be
    possible to use the provided `N` as output.

See also [`right_null(!)`](@ref right_null)
"""
@functiondef left_null

"""
    right_null(A; atol::Real, rtol::Real) -> N
    right_null(A, alg::AbstractAlgorithm) -> N
    right_null!(A, [N]; atol::Real, rtol::Real) -> N
    right_null!(A, [N], alg::AbstractAlgorithm) -> N

Compute the right nullspace of the matrix `A` of size `(m, n)` with `m ≤ n`,
such that `N * A ≈ 0`. Here, `N` is a matrix with orthonormal columns that span the
kernel of `A`. By default, this is achieved by including the right singular vectors
whose singular values have magnitude smaller than `max(atol, rtol * σ₁)`, where `σ₁`
is `A`'s largest singular value.

The keyword arguments can be used to control the precision and have default values
`atol=0` and `rtol=atol > 0 ? 0 : n*ϵ`, where `n` is the size of the smallest dimension of `A`,
and `ϵ` is the `eps` of the element type.

!!! note
    The bang method `right_null!` optionally accepts the output structure and possibly destroys
    the input matrix `A`. Always use the return value of the function as it may not always be
    possible to use the provided `N` as output.

See also [`left_null(!)`](@ref left_null).
"""
@functiondef right_null

# Algorithm selection
# -------------------
for f in (:left_null, :right_null)
    f! = Symbol(f, :!)
    @eval begin
        function select_algorithm(::typeof($f), A; kwargs...)
            return select_algorithm($f!, A; kwargs...)
        end
        function select_algorithm(::typeof($f!), A; alg=nothing, atol=nothing, rtol=nothing)
            if alg isa TruncatedAlgorithm
                (isnothing(atol) && isnothing(rtol)) ||
                    @warn "User supplied algorithm and kwargs"
                return alg
            else
                alg_svd = select_algorithm(svd_full!, A; alg)
                atol′ = something(atol, 0)
                rtol′ = something(rtol,
                                  atol′ > 0 ? 0 :
                                  minimum(size(A)) * eps(real(float(eltype(A)))))
                trunc = TruncationKeepBelow(atol′, rtol′)
                return TruncatedAlgorithm(alg_svd, trunc)
            end
        end
    end
end
