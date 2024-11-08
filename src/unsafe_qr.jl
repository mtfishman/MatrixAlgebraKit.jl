# Define in-place QR decomposition with optional pivoting and positive diagonal of R.
# This is the method that should be overloaded for in package extensions, for other scalar or matrix types.

function _unsafe_qr!(A::StridedMatrix{S}, Q::StridedMatrix{S}, R::StridedMatrix{S};
                     positive=false,
                     pivoted=false,
                     blocksize=((pivoted || A === Q) ? 1 : YALAPACK.default_qr_blocksize(A))) where {S<:BlasFloat}
    m, n = size(A)
    minmn = min(m, n)
    computeR = length(R) > 0
    inplaceQ = Q === A

    if pivoted && (blocksize > 1)
        throw(ArgumentError("LAPACK does not provide a blocked implementation for a pivoted QR decomposition"))
    end
    if inplaceQ && (computeR || positive || blocksize > 1 || m < n)
        throw(ArgumentError("inplace Q only supported if matrix is tall (`m >= n`), R is not required, and using the unblocked algorithm (`blocksize=1`) with `positive=false`"))
    end

    if blocksize > 1
        nb = min(minmn, blocksize)
        if computeR # first use R as space for T
            A, T = YALAPACK.geqrt!(A, view(R, 1:nb, 1:minmn))
        else
            A, T = YALAPACK.geqrt!(A, similar(A, nb, minmn))
        end
        Q = YALAPACK.gemqrt!('L', 'N', A, T, one!(Q))
    else
        if pivoted
            A, τ, jpvt = YALAPACK.geqp3!(A)
        else
            A, τ = YALAPACK.geqrf!(A)
        end
        if inplaceQ
            Q = YALAPACK.orgqr!(A, τ)
        else
            Q = YALAPACK.ormqr!('L', 'N', A, τ, one!(Q))
        end
    end

    if positive # already fix Q even if we do not need R
        @inbounds for j in 1:minmn
            s = safesign(A[j, j])
            @simd for i in 1:m
                Q[i, j] *= s
            end
        end
    end

    if computeR
        R̃ = triu!(view(A, axes(R)...))
        if positive
            @inbounds for j in n:-1:1
                @simd for i in 1:min(minmn, j)
                    R̃[i, j] = R̃[i, j] * conj(safesign(R̃[i, i]))
                end
            end
        end
        if !pivoted
            copyto!(R, R̃)
        else
            # probably very inefficient in terms of memory access
            copyto!(view(R, :, jpvt), R̃)
        end
    end
    return Q, R
end
