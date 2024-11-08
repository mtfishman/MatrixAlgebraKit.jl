# TODO: Decomposition or factorisation?
# Decomposition has no chance of colliding with Base factorisation types

abstract type Decomposition end
abstract type OrthogonalDecomposition <: Decomposition end

for D in (:QRDecomposition, :QLDecomposition, :RQDecomposition, :LQDecomposition)
    @eval begin
        @kwdef struct $D{A} <: OrthogonalDecomposition
            positive::Bool = false
            pivoted::Bool = false
            alg::A = nothing
        end
    end

    function $D(A::AbstractMatrix)
        return $D(; default_qr_parameters(A)...)
    end
end

struct SingularValueDecomposition{A} <: OrthogonalDecomposition
    svdalg::{A}
end

struct PolarDecomposition <: OrthogonalDecomposition
    svdalg::{A}
end

function SingularValueDecomposition(A::AbstractMatrix)
    return SingularValueDecomposition(default_svd_alg(A))
end
PolarDecomposition(A::AbstractMatrix) = PolarDecomposition(default_svd_alg(A))

Base.adjoint(d::QRDecomposition) = LQDecomposition(d.positive, d.pivoted, d.alg)
Base.adjoint(d::LQDecomposition) = QRDecomposition(d.positive, d.pivoted, d.alg)
Base.adjoint(d::QLDecomposition) = RQDecomposition(d.positive, d.pivoted, d.alg)
Base.adjoint(d::RQDecomposition) = QLDecomposition(d.positive, d.pivoted, d.alg)

Base.adjoint(d::SingularValueDecomposition) = d
Base.adjoint(d::PolarDecomposition) = d

function default_svd_alg end

function default_svd_alg(A::AbstractMatrix{T}) where {T<:BlasFloat}
    return LinearAlgebra.default_svd_alg(A)
end
