# TODO: export? or not export but mark as public ?
function schur(A::AbstractMatrix, args...; kwargs...)
    return schur_full!(A, args...; kwargs...)
end

# copy input
function copy_input(::typeof(schur_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(schur_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

# initialize output
function initialize_output(::typeof(schur_full!), A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    Z = similar(A, (n, n))
    vals = similar(A, complex(eltype(A)), n)
    return (A, Z, vals)
end
function initialize_output(::typeof(schur_vals!), A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    vals = similar(A, complex(eltype(A)), n)
    return vals
end
function schur_vals_init(A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    vals = similar(A, complex(eltype(A)), n)
    return vals
end

# select default algorithm
function select_algorithm(::typeof(schur_full!), A::AbstractMatrix; kwargs...)
    return default_schur_algorithm(A; kwargs...)
end
function select_algorithm(::typeof(schur_vals!), A::AbstractMatrix; kwargs...)
    return default_schur_algorithm(A; kwargs...)
end

function default_schur_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_Expert(; kwargs...)
end

# check input
function check_input(::typeof(schur_full!), A::AbstractMatrix, TZv)
    m, n = size(A)
    m == n || throw(ArgumentError("Schur decompsition requires square input matrix"))
    T, Z, vals = TZv
    (Z isa AbstractMatrix && eltype(Z) == eltype(A) && size(Z) == (m, m)) ||
        throw(ArgumentError("`schur_full!` requires square Z matrix with same size and `eltype` as A"))
    (T isa AbstractMatrix && eltype(T) == eltype(A) && size(T) == (m, m)) ||
        throw(ArgumentError("`schur_full!` requires square T matrix with same size and `eltype` as A"))
    size(vals) == (n,) && eltype(vals) == complex(eltype(A)) ||
        throw(ArgumentError("Eigenvalue vector `vals` must have length equal to size(A, 1) and complex `eltype`"))
    return nothing
end
function check_input(::typeof(schur_vals!), A::AbstractMatrix, vals)
    m, n = size(A)
    m == n || throw(ArgumentError("Schur decompsition requires square input matrix"))
    size(vals) == (n,) && eltype(vals) == complex(eltype(A)) ||
        throw(ArgumentError("Eigenvalue vector `vals` must have length equal to size(A, 1) and complex `eltype`"))
    return nothing
end

# actual implementation
function schur_full!(A::AbstractMatrix, TZv, alg::LAPACK_EigAlgorithm)
    check_input(schur_full!, A, TZv)
    T, Z, vals = TZv
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple does not accept any keyword arguments"))
        YALAPACK.gees!(A, Z, vals)
    else # alg isa LAPACK_Expert
        YALAPACK.geesx!(A, Z, vals; alg.kwargs...)
    end
    T === A || copy!(T, A)
    return T, Z, vals
end

function schur_vals!(A::AbstractMatrix, vals, alg::LAPACK_EigAlgorithm)
    check_input(schur_vals!, A, vals)
    Z = similar(A, eltype(A), (size(A, 1), 0))
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple does not accept any keyword arguments"))
        YALAPACK.gees!(A, Z, vals)
    else # alg isa LAPACK_Expert
        YALAPACK.geesx!(A, Z, vals; alg.kwargs...)
    end
    return vals
end

# # for schur_trunc!, it doesn't make sense to preallocate D and V as we don't know their sizes
# function select_algorithm(::typeof(schur_trunc!), A::AbstractMatrix; kwargs...)
#     return default_schur_algorithm(A; kwargs...)
# end

# function schur_trunc!(A::AbstractMatrix, alg::LAPACK_EigAlgorithm, trunc::TruncationStrategy)
#     DV = schur_full_init(A)
#     D, V = schur_full!(A, DV, alg)

#     Dd = D.diag
#     ind = findtruncated(Dd, trunc)
#     V′ = V[:, ind]
#     D′ = Diagonal(Dd[ind])
#     return (D′, V′)
# end
