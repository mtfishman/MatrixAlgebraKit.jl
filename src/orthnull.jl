# orthnull.jl
# methods for computing an orthonormal basis for the image or null space of a matrix or its adjoint

# function _check_leftorthargs(A, Q, R, alg::Union{QRDecomposition,QLDecomposition})
#     m, n = size(A)
#     m == size(Q, 1) || throw(DimensionMismatch("size mismatch between A and Q"))
#     n == size(R, 2) || throw(DimensionMismatch("size mismatch between A and R"))
#     size(Q, 2) <= m || throw(DimensionMismatch("Q has more columns than rows"))
#     k = min(m, n)
#     size(Q, 2) >= k || @warn "Q has too few columns, truncating QR factorization"
#     if length(R) > 0
#         size(Q, 2) == size(R, 1) ||
#             throw(DimensionMismatch("size mismatch between Q and R"))
#         if alg isa Union{QL,QLpos}
#             size(R, 1) == n ||
#                 throw(DimensionMismatch("QL factorisation requires square R"))
#         end
#     end
#     return m, n, k
# end

# actual implementation, can be overloaded for specific matrix types

# function leftorth!(A::StridedMatrix{S}, Q::StridedMatrix{S}, R::StridedMatrix{S},
#                    decomposition::QRDecomposition;
#                    atol::Real=zero(real(S))) where {S<:BlasFloat}
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n, k = _check_leftorthargs(A, Q, R, alg)

#     A, T = LAPACK.geqrt!(A, min(k, 36))
#     Q = LAPACK.gemqrt!('L', 'N', A, T, one!(Q))
#     if length(R) > 0
#         R = copy!(R, triu!(view(A, axes(R)...)))
#         if decomposition.positive == true
#             @inbounds for j in 1:k
#                 s = safesign(R[j, j])
#                 @simd for i in 1:m
#                     Q[i, j] *= s
#                 end
#             end
#             @inbounds for j in size(R, 2):-1:1
#                 for i in 1:min(k, j)
#                     R[i, j] = R[i, j] * conj(safesign(R[i, i]))
#                 end
#             end
#         end
#     end
#     return Q, R
# end

# function leftorth!(A::StridedMatrix{S}, Q::StridedMatrix{S}, R::StridedMatrix{S},
#                    alg::Union{QL,QLpos}, atol::Real) where {S<:BlasFloat}
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n, k = _check_leftorthargs(A, Q, R, alg)

#     nhalf = div(n, 2)
#     #swap columns in A
#     @inbounds for j in 1:nhalf, i in 1:m
#         A[i, j], A[i, n + 1 - j] = A[i, n + 1 - j], A[i, j]
#     end

#     # perform QR factorization
#     leftorth!(A, Q, R, isa(alg, QL) ? QR() : QRpos(), atol)

#     #swap columns in Q
#     @inbounds for j in 1:nhalf, i in 1:m
#         Q[i, j], Q[i, n + 1 - j] = Q[i, n + 1 - j], Q[i, j]
#     end
#     #swap rows and columns in R
#     @inbounds for j in 1:nhalf, i in 1:n
#         R[i, j], R[n + 1 - i, n + 1 - j] = R[n + 1 - i, n + 1 - j], R[i, j]
#     end
#     if isodd(n)
#         j = nhalf + 1
#         @inbounds for i in 1:nhalf
#             R[i, j], R[n + 1 - i, j] = R[n + 1 - i, j], R[i, j]
#         end
#     end
#     return Q, R
# end

# function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A) : LAPACK.gesdd!('S', A)
#     if isa(alg, Union{SVD,SDD})
#         n = count(s -> s .> atol, S)
#         if n != length(S)
#             return U[:, 1:n], lmul!(Diagonal(S[1:n]), V[1:n, :])
#         else
#             return U, lmul!(Diagonal(S), V)
#         end
#     else
#         iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#         # TODO: check Lapack to see if we can recycle memory of A
#         Q = mul!(A, U, V)
#         Sq = map!(sqrt, S, S)
#         SqV = lmul!(Diagonal(Sq), V)
#         R = SqV' * SqV
#         return Q, R
#     end
# end

# function leftnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos}, atol::Real)
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n = size(A)
#     m >= n || throw(ArgumentError("no null space if less rows than columns"))

#     A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
#     N = similar(A, m, max(0, m - n))
#     fill!(N, 0)
#     for k in 1:(m - n)
#         N[n + k, k] = 1
#     end
#     return N = LAPACK.gemqrt!('L', 'N', A, T, N)
# end

# function leftnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
#     size(A, 2) == 0 && return one!(similar(A, (size(A, 1), size(A, 1))))
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('A', 'N', A) : LAPACK.gesdd!('A', A)
#     indstart = count(>(atol), S) + 1
#     return U[:, indstart:end]
# end

# function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos,RQ,RQpos},
#                     atol::Real)
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     # TODO: geqrfp seems a bit slower than geqrt in the intermediate region around
#     # matrix size 100, which is the interesting region. => Investigate and fix
#     m, n = size(A)
#     k = min(m, n)
#     At = transpose!(similar(A, n, m), A)

#     if isa(alg, RQ) || isa(alg, RQpos)
#         @assert m <= n

#         mhalf = div(m, 2)
#         # swap columns in At
#         @inbounds for j in 1:mhalf, i in 1:n
#             At[i, j], At[i, m + 1 - j] = At[i, m + 1 - j], At[i, j]
#         end
#         Qt, Rt = leftorth!(At, isa(alg, RQ) ? QR() : QRpos(), atol)

#         @inbounds for j in 1:mhalf, i in 1:n
#             Qt[i, j], Qt[i, m + 1 - j] = Qt[i, m + 1 - j], Qt[i, j]
#         end
#         @inbounds for j in 1:mhalf, i in 1:m
#             Rt[i, j], Rt[m + 1 - i, m + 1 - j] = Rt[m + 1 - i, m + 1 - j], Rt[i, j]
#         end
#         if isodd(m)
#             j = mhalf + 1
#             @inbounds for i in 1:mhalf
#                 Rt[i, j], Rt[m + 1 - i, j] = Rt[m + 1 - i, j], Rt[i, j]
#             end
#         end
#         Q = transpose!(A, Qt)
#         R = transpose!(similar(A, (m, m)), Rt) # TODO: efficient in place
#         return R, Q
#     else
#         Qt, Lt = leftorth!(At, alg', atol)
#         if m > n
#             L = transpose!(A, Lt)
#             Q = transpose!(similar(A, (n, n)), Qt) # TODO: efficient in place
#         else
#             Q = transpose!(A, Qt)
#             L = transpose!(similar(A, (m, m)), Lt) # TODO: efficient in place
#         end
#         return L, Q
#     end
# end

# function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A) : LAPACK.gesdd!('S', A)
#     if isa(alg, Union{SVD,SDD})
#         n = count(s -> s .> atol, S)
#         if n != length(S)
#             return rmul!(U[:, 1:n], Diagonal(S[1:n])), V[1:n, :]
#         else
#             return rmul!(U, Diagonal(S)), V
#         end
#     else
#         iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#         Q = mul!(A, U, V)
#         Sq = map!(sqrt, S, S)
#         USq = rmul!(U, Diagonal(Sq))
#         L = USq * USq'
#         return L, Q
#     end
# end

# function rightnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos}, atol::Real)
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n = size(A)
#     k = min(m, n)
#     At = adjoint!(similar(A, n, m), A)
#     At, T = LAPACK.geqrt!(At, min(k, 36))
#     N = similar(A, max(n - m, 0), n)
#     fill!(N, 0)
#     for k in 1:(n - m)
#         N[k, m + k] = 1
#     end
#     return N = LAPACK.gemqrt!('R', eltype(At) <: Real ? 'T' : 'C', At, T, N)
# end

# function rightnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
#     size(A, 1) == 0 && return one!(similar(A, (size(A, 2), size(A, 2))))
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('N', 'A', A) : LAPACK.gesdd!('A', A)
#     indstart = count(>(atol), S) + 1
#     return V[indstart:end, :]
# end
