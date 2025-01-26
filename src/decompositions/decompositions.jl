# TODO: module Decompositions?

const _DECOMPOSITION_LIST = (:qr_full, :qr_compact,
                             :schur_full, :schur_vals)

function copy_input end
function initialize_output end
function check_input end

for f in _DECOMPOSITION_LIST
    @eval @functiondef $f
    # f! = Symbol(f, :!)
    # @eval begin
    #     function $f(A; kwargs...)
    #         Ac = copy_input($f, A)
    #         return $f!(Ac; kwargs...)
    #     end
    #     function $f(A, alg::Algorithm)
    #         Ac = copy_input($f, A)
    #         return $f!(Ac, alg)
    #     end
    #     function $f!(A; kwargs...)
    #         out = initialize_output($f!, A)
    #         return $f!(A, out; kwargs...)
    #     end
    #     function $f!(A, out; kwargs...)
    #         alg = select_algorithm($f!, A; kwargs...)
    #         return $f!(A, out, alg)
    #     end
    #     function $f!(A, alg::Algorithm)
    #         out = initialize_output($f!, A)
    #         return $f!(A, out, alg)
    #     end
    # end
end

# ==========
# ALGORITHMS
# ==========

# reference for naming LAPACK algorithms:
# https://www.netlib.org/lapack/explore-html/modules.html

# QR Decomposition
@algdef LAPACK_HouseholderQR

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
