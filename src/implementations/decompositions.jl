# TODO: module Decompositions?

# ==========
# ALGORITHMS
# ==========

# reference for naming LAPACK algorithms:
# https://www.netlib.org/lapack/explore-html/topics.html

# QR, LQ, QL, RQ Decomposition
@algdef LAPACK_HouseholderQR
@algdef LAPACK_HouseholderLQ
@algdef LAPACK_HouseholderQL
@algdef LAPACK_HouseholderRQ

# General Eigenvalue Decomposition
@algdef LAPACK_Simple
@algdef LAPACK_Expert

const LAPACK_EigAlgorithm = Union{LAPACK_Simple,LAPACK_Expert}

# Hermitian Eigenvalue Decomposition
const LAPACK_QRIteration = LAPACK_Simple
export LAPACK_QRIteration
const LAPACK_Bisection = LAPACK_Expert
export LAPACK_Bisection
@algdef LAPACK_DivideAndConquer
@algdef LAPACK_MultipleRelativelyRobustRepresentations

const LAPACK_EighAlgorithm = Union{LAPACK_QRIteration,
                                   LAPACK_Bisection,
                                   LAPACK_DivideAndConquer,
                                   LAPACK_MultipleRelativelyRobustRepresentations}

# Singular Value Decomposition
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
