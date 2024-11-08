function qr_full!(A::AbstractMatrix,
                  Q::AbstractMatrix=similar(A, (size(A, 1), size(A, 1))),
                  R::AbstractMatrix=similar(A, (size(A, 1), size(A, 2)));
                  kwargs...)
    return qr_full!(A, Q, R, default_backend(qr_full!, A; kwargs...))
end
function qr_compact!(A::AbstractMatrix,
                     Q::AbstractMatrix=similar(A, (size(A, 1), size(A, 1))),
                     R::AbstractMatrix=similar(A, (size(A, 1), size(A, 2)));
                     kwargs...)
    return qr_compact!(A, Q, R, default_backend(qr_compact!, A; kwargs...))
end

function default_backend(::typeof(qr_full!), A::AbstractMatrix; kwargs...)
    return default_qr_backend(A; kwargs...)
end
function default_backend(::typeof(qr_compact!), A::AbstractMatrix; kwargs...)
    return default_qr_backend(A; kwargs...)
end

function default_qr_backend(A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    return LAPACKBackend()
end