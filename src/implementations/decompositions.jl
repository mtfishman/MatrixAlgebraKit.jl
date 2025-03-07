# TODO: module Decompositions?

# ==========
# ALGORITHMS
# ==========

# reference for naming LAPACK algorithms:
# https://www.netlib.org/lapack/explore-html/topics.html

# QR, LQ, QL, RQ Decomposition
"""
    LAPACK_HoudeholderQR(; blocksize, positive = false, pivoted = false)

Algorithm type to denote the standard LAPACK algorithm for computing the
QR decomposition of a matrix using Householder reflectors. The specific
LAPACK function can be controlled using the keyword arugments, i.e.
`?geqrt` will be chosen if `blocksize > 1`. With `blocksize == 1`,
`?geqrf` will be chosen if `pivoted == false` and
`?geqp3` will be chosen if `pivoted == true`.
"""
@algdef LAPACK_HouseholderQR

"""
    LAPACK_HoudeholderLQ(; blocksize, positive = false)

Algorithm type to denote the standard LAPACK algorithm for computing the
LQ decomposition of a matrix using Householder reflectors. The specific
LAPACK function can be controlled using the keyword arugments, i.e.
`?gelqt` will be chosen if `blocksize > 1` or
`?gelqf` will be chosen if `pivoted == false`.
"""
@algdef LAPACK_HouseholderLQ
@algdef LAPACK_HouseholderQL
@algdef LAPACK_HouseholderRQ

# General Eigenvalue Decomposition
"""
    LAPACK_Simple()

Algorithm type to denote the simple LAPACK driver for computing the
Schur or nonhermitian eigenvalue decomposition of a matrix. For the
Hermitian eigenvalue decomposition, this coincides with using the
QR iteration algorithm.
"""
@algdef LAPACK_Simple

"""
    LAPACK_Expert()

Algorithm type to denote the expert LAPACK driver for computing the
Schur or nonhermitian eigenvalue decomposition of a matrix. For the
Hermitian eigenvalue decomposition, this coincides with the
bisection algorithm.
"""
@algdef LAPACK_Expert

const LAPACK_EigAlgorithm = Union{LAPACK_Simple,LAPACK_Expert}

# Hermitian Eigenvalue Decomposition
"""
    LAPACK_QRIteration()

Algorithm type to denote the LAPACK driver for computing
the eigenvalue decomposition of a Hermitian matrix,
or the singular value decomposition of a general matrix
using the QR Iteration algorithm.
"""
const LAPACK_QRIteration = LAPACK_Simple
# TODO: currently have two docstrings, should we keep both?

export LAPACK_QRIteration

"""
    LAPACK_Bisection()

Algorithm type to denote the LAPACK driver for computing
the eigenvalue decomposition of a Hermitian matrix,
or the singular value decomposition of a general matrix
using the Bisection algorithm.
"""
const LAPACK_Bisection = LAPACK_Expert
# TODO: currently have two docstrings, should we keep both?

export LAPACK_Bisection

"""
    LAPACK_DivideAndConquer()

Algorithm type to denote the LAPACK driver for computing
the eigenvalue decomposition of a Hermitian matrix,
or the singular value decomposition of a general matrix
using the Divide and Conquer algorithm.
"""
@algdef LAPACK_DivideAndConquer

"""
    LAPACK_MultipleRelativelyRobustRepresentations()

Algorithm type to denote the LAPACK driver for computing
the eigenvalue decomposition of a Hermitian matrix
using the Multiple Relatively Robust Representations algorithm
"""
@algdef LAPACK_MultipleRelativelyRobustRepresentations

const LAPACK_EighAlgorithm = Union{LAPACK_QRIteration,
                                   LAPACK_Bisection,
                                   LAPACK_DivideAndConquer,
                                   LAPACK_MultipleRelativelyRobustRepresentations}

# Singular Value Decomposition
"""
    LAPACK_Jacobi()

Algorithm type to denote the LAPACK driver for computing
the singular value decomposition of a general matrix
using the Jacobi algorithm
"""
@algdef LAPACK_Jacobi

const LAPACK_SVDAlgorithm = Union{LAPACK_QRIteration,
                                  LAPACK_Bisection,
                                  LAPACK_DivideAndConquer,
                                  LAPACK_Jacobi}

# TODO: Decomposition or factorisation?
# Decomposition has no chance of colliding with Base factorisation types

# OLD STUFF

# abstract type Decomposition end
# abstract type OrthogonalDecomposition <: Decomposition end

# for D in (:QRDecomposition, :QLDecomposition, :RQDecomposition, :LQDecomposition)
#     @eval begin
#         @kwdef struct $D{A} <: OrthogonalDecomposition
#             positive::Bool = false
#             pivoted::Bool = false
#             alg::A = nothing
#         end
#     end

#     function $D(A::AbstractMatrix)
#         return $D(; default_qr_parameters(A)...)
#     end
# end

# struct SingularValueDecomposition{A} <: OrthogonalDecomposition
#     svdalg::{A}
# end

# struct PolarDecomposition <: OrthogonalDecomposition
#     svdalg::{A}
# end

# function SingularValueDecomposition(A::AbstractMatrix)
#     return SingularValueDecomposition(default_svd_alg(A))
# end
# PolarDecomposition(A::AbstractMatrix) = PolarDecomposition(default_svd_alg(A))

# Base.adjoint(d::QRDecomposition) = LQDecomposition(d.positive, d.pivoted, d.alg)
# Base.adjoint(d::LQDecomposition) = QRDecomposition(d.positive, d.pivoted, d.alg)
# Base.adjoint(d::QLDecomposition) = RQDecomposition(d.positive, d.pivoted, d.alg)
# Base.adjoint(d::RQDecomposition) = QLDecomposition(d.positive, d.pivoted, d.alg)

# Base.adjoint(d::SingularValueDecomposition) = d
# Base.adjoint(d::PolarDecomposition) = d

# function default_svd_alg end

# function default_svd_alg(A::AbstractMatrix{T}) where {T<:BlasFloat}
#     return LinearAlgebra.default_svd_alg(A)
# end
