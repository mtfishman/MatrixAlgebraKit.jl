# Inputs
# ------
function copy_input(::typeof(eig_full), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
function copy_input(::typeof(eig_vals), A::AbstractMatrix)
    return copy!(similar(A, float(eltype(A))), A)
end
copy_input(::typeof(eig_trunc), A) = copy_input(eig_full, A)

function check_input(::typeof(eig_full!), A::AbstractMatrix, DV)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decomposition requires square input matrix"))
    D, V = DV
    Tc = complex(eltype(A))
    (V isa AbstractMatrix && eltype(V) == Tc && size(V) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires square matrix V with same size as A and complex `eltype`"))
    (D isa Diagonal && eltype(D) == Tc && size(D) == (m, m)) ||
        throw(ArgumentError("`eig_full!` requires Diagonal matrix D with same size as A and complex `eltype`"))
    return nothing
end
function check_input(::typeof(eig_vals!), A::AbstractMatrix, D)
    m, n = size(A)
    m == n || throw(ArgumentError("Eigenvalue decomposition requires square input matrix"))
    Tc = complex(eltype(A))
    size(D) == (n,) && eltype(D) == Tc ||
        throw(ArgumentError("Eigenvalue vector `D` must have length equal to size(A, 1) and complex `eltype`"))
    return nothing
end

# Outputs
# -------
function initialize_output(::typeof(eig_full!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = Diagonal(similar(A, Tc, n))
    V = similar(A, Tc, (n, n))
    return (D, V)
end
function initialize_output(::typeof(eig_vals!), A::AbstractMatrix, ::LAPACK_EigAlgorithm)
    n = size(A, 1) # square check will happen later
    Tc = complex(eltype(A))
    D = similar(A, Tc, n)
    return D
end
function initialize_output(::typeof(eig_trunc!), A::AbstractMatrix, alg::TruncatedAlgorithm)
    return initialize_output(eig_full!, A, alg.alg)
end

# Implementation
# --------------
# actual implementation
function eig_full!(A::AbstractMatrix, DV, alg::LAPACK_EigAlgorithm)
    check_input(eig_full!, A, DV)
    D, V = DV
    if alg isa LAPACK_Simple
        isempty(alg.kwargs) ||
            throw(ArgumentError("LAPACK_Simple (geev) does not accept any keyword arguments"))
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
            throw(ArgumentError("LAPACK_Simple (geev) does not accept any keyword arguments"))
        YALAPACK.geev!(A, D, V)
    else # alg isa LAPACK_Expert
        YALAPACK.geevx!(A, D, V; alg.kwargs...)
    end
    return D
end

function eig_trunc!(A::AbstractMatrix, DV, alg::TruncatedAlgorithm)
    D, V = eig_full!(A, DV, alg.alg)
    ind = findtruncated(diagview(D), alg.trunc)
    return truncate!((D, V), ind)
end
