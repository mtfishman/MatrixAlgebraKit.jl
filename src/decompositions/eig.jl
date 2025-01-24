# TODO: export? or not export but mark as public ?
function eig!(A::AbstractMatrix, args...; kwargs...)
    return eig_full!(A, args...; kwargs...)
end

# copy input
function copy_input(::typeof(eig_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(eig_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end

# initialize output
function initialize_output(::typeof(eig_full!), A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = Diagonal(similar(A, Tc, n))
    V = similar(A, Tc, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eig_vals!), A::AbstractMatrix)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = similar(A, Tc, n)
    return D
end

# select default algorithm
function default_algorithm(::typeof(eig_full!), A::AbstractMatrix; kwargs...)
    return default_eig_algorithm(A; kwargs...)
end
function default_algorithm(::typeof(eig_vals!), A::AbstractMatrix; kwargs...)
    return default_eig_algorithm(A; kwargs...)
end

function default_eig_algorithm(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACK_Expert(; kwargs...)
end

# check input
function check_input(::typeof(eig_full!), A::AbstractMatrix, DV)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    D, V = DV
    Tc = complex(eltype(A))
    (V isa AbstractMatrix && eltype(V) == Tc && size(V) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires square V matrix with same size as A and complex `eltype`"))
    (D isa Diagonal && eltype(D) == Tc && size(D) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires Diagonal matrix D with same size as A and complex `eltype`"))
    return nothing
end
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decompsition requires square input matrix"))
    Tc = complex(eltype(A))
    size(D) == (n,) && eltype(D) == Tc ||
        throw(ArgumentError("Eigenvalue vector `D` must have length equal to size(A, 1) and complex `eltype`"))
    return nothing
end

# actual implementation
function eig_full!(A::AbstractMatrix, DV, alg::LAPACK_EigAlgorithm)
    check_input(eig_full!, A, DV)
    D, V = DV
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple does not accept any keyword arguments"))
        YALAPACK.geev!(A, D.diag, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D.diag, V; alg.kwargs...)
    end
    return D, V
end
function eig_vals!(A::AbstractMatrix, D, alg::LAPACK_EigAlgorithm)
    check_input(eig_vals!, A, D)
    V = similar(A, complex(eltype(A)), (size(A, 1), 0))
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple does not accept any keyword arguments"))
        YALAPACK.geev!(A, D, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D, V; alg.kwargs...)
    end
    return D, V
end

# # for eig_trunc!, it doesn't make sense to preallocate D and V as we don't know their sizes
# function default_algorithm(::typeof(eig_trunc!), A::AbstractMatrix; kwargs...)
#     return default_eig_algorithm(A; kwargs...)
# end

# function eig_trunc!(A::AbstractMatrix, alg::LAPACK_EigAlgorithm, trunc::TruncationStrategy)
#     DV = eig_full_init(A)
#     D, V = eig_full!(A, DV, alg)

#     Dd = D.diag
#     ind = findtruncated(Dd, trunc)
#     V′ = V[:, ind]
#     D′ = Diagonal(Dd[ind])
#     return (D′, V′)
# end