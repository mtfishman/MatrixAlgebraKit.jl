function one!(A::StridedMatrix)
    length(A) > 0 || return A
    copyto!(A, LinearAlgebra.I)
    return A
end

safesign(s::Real) = ifelse(s < zero(s), -one(s), +one(s))
safesign(s::Complex) = ifelse(iszero(s), one(s), s / abs(s))

diagview(D::Diagonal) = D.diag
diagview(D::AbstractMatrix) = view(D, diagind(D))
