# TODO: Consider using zerovector! if using VectorInterface.jl
function zero!(A::AbstractMatrix)
    A .= zero(eltype(A))
    return A
end

function one!(A::AbstractMatrix)
    length(A) > 0 || return A
    copyto!(A, LinearAlgebra.I)
    return A
end
