function one!(A::StridedMatrix)
    length(A) > 0 || return A
    copyto!(A, LinearAlgebra.I)
    return A
end

safesign(s::Real) = ifelse(s < zero(s), -one(s), +one(s))
safesign(s::Complex) = ifelse(iszero(s), one(s), s / abs(s))

safe_inv(a, tol) = abs(a) < tol ? zero(a) : inv(a)

# diagind: provided by LinearAlgebra.jl

function lowertriangularind(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    I = Vector{Int}(undef, div(m * (m - 1), 2) + m * (n - m))
    offset = 0
    for j in 1:n
        r = (j + 1):m
        I[offset .- j .+ r] = (j - 1) * m .+ r
        offset += length(r)
    end
    return I
end

function uppertriangularind(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    I = Vector{Int}(undef, div(m * (m - 1), 2) + m * (n - m))
    offset = 0
    for i in 1:m
        r = (i + 1):n
        I[offset .- i .+ r] = i .+ m .* (r .- 1)
        offset += length(r)
    end
    return I
end

diagview(D::Diagonal) = D.diag
diagview(D::AbstractMatrix) = view(D, diagind(D))
