module YALAPACK # Yet another lapack wrapper

using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, Char, LAPACK,
                     LAPACKException, SingularException, PosDefException,
                     checksquare, chkstride1, require_one_based_indexing, triu!,
                     issymmetric, ishermitian, isposdef, adjoint!

using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using LinearAlgebra.LAPACK: chkfinite, chktrans, chkside, chkuplofinite, chklapackerror

# LU factorisation
for (getrf, getrs, elty) in ((:dgetrf_, :dgetrs_, :Float64),
                             (:sgetrf_, :sgetrs_, :Float32),
                             (:zgetrf_, :zgetrs_, :ComplexF64),
                             (:cgetrf_, :cgetrs_, :ComplexF32))
    @eval begin
        function getrf!(A::AbstractMatrix{$elty}, ipiv::AbstractVector{BlasInt};
                        check::Bool=true)
            require_one_based_indexing(A, ipiv)
            chkstride1(A, ipiv)
            chkfinite(A)
            m, n = size(A)

            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrf), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, ipiv, info)
            chkargsok(info[])
            return A, ipiv, info[] #Error code is stored in LU factorization type
        end
        function getrs!(trans::AbstractChar, A::AbstractMatrix{$elty},
                        ipiv::AbstractVector{BlasInt}, B::AbstractVecOrMat{$elty})
            require_one_based_indexing(A, ipiv, B)
            chktrans(trans)
            chkstride1(A, B, ipiv)
            n = checksquare(A)
            if n != size(B, 1)
                throw(DimensionMismatch(lazy"B has leading dimension $(size(B,1)), but needs $n"))
            end
            if n != length(ipiv)
                throw(DimensionMismatch(lazy"ipiv has length $(length(ipiv)), but needs to be $n"))
            end
            nrhs = size(B, 2)
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrs), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
                  trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B,
                  max(1, stride(B, 2)), info, 1)
            chklapackerror(info[])
            return B
        end
    end
end

# QR factorisation
const DEFAULT_QR_BLOCKSIZE = 36
default_qr_blocksize(A::AbstractMatrix) = min(size(A)..., DEFAULT_QR_BLOCKSIZE)

#! format: off
for (gelqf, gelqt, geqlf, geqrf, geqp3, geqrt, gerqf, elty, relty) in
    ((:dgelqf_, :dgelqt_, :dgeqlf_, :dgeqrf_, :dgeqp3_, :dgeqrt_, :dgerqf_, :Float64, :Float64),
     (:sgelqf_, :sgelqt_, :sgeqlf_, :sgeqrf_, :sgeqp3_, :sgeqrt_, :sgerqf_, :Float32, :Float32),
     (:zgelqf_, :zgelqt_, :zgeqlf_, :zgeqrf_, :zgeqp3_, :zgeqrt_, :zgerqf_, :ComplexF64, :Float64),
     (:cgelqf_, :cgelqt_, :cgeqlf_, :cgeqrf_, :cgeqp3_, :cgeqrt_, :cgerqf_, :ComplexF32, :Float32))
#! format: on
    @eval begin
        # QR with block reflectors
        #! format: off
        function geqrt!(A::AbstractMatrix{$elty},
                        T::AbstractMatrix{$elty}=similar(A, $elty, default_qr_blocksize(A), min(size(A)...)))
        #! format: on
            require_one_based_indexing(A, T)
            chkstride1(A)
            m, n = size(A)
            minmn = min(m, n)
            nb = size(T, 1)
            nb <= minmn ||
                throw(ArgumentError(lazy"block size $nb > $minmn too large"))
            size(T, 2) == minmn ||
                throw(DimensionMismatch(lazy"block reflector T should have size ($nb,$minmn)"))
            n == 0 && return A, T

            lda = max(1, stride(A, 2))
            ldt = max(1, stride(T, 2))
            work = Vector{$elty}(undef, nb * n)
            if minmn > 0
                info = Ref{BlasInt}()
                ccall((@blasfunc($geqrt), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{BlasInt}),
                      m, n, nb, A, lda,
                      T, ldt, work,
                      info)
                chklapackerror(info[])
            end
            return A, T
        end
        # QR
        function geqrf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i in 1:2                # first call returns lwork as work[1]
                ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = max(BlasInt(1), BlasInt(real(work[1])))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
        # QR with column pivoting
        function geqp3!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)),
                        jpvt::AbstractVector{BlasInt}=zeros(BlasInt, size(A, 2)))
            require_one_based_indexing(A, jpvt, tau)
            chkstride1(A, jpvt, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            length(jpvt) == n ||
                throw(DimensionMismatch(lazy"jpvt has length $(length(jpvt)), but needs length $n"))
            n == 0 && return A, tau, jpvt

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            if cmplx
                rwork = Vector{$relty}(undef, 2n)
            end
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                if cmplx
                    ccall((@blasfunc($geqp3), libblastrampoline), Cvoid,
                          (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{$elty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}),
                          m, n, A, lda,
                          jpvt, tau,
                          work, lwork, rwork,
                          info)
                else
                    ccall((@blasfunc($geqp3), libblastrampoline), Cvoid,
                          (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{$elty},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}),
                          m, n, A, lda,
                          jpvt, tau,
                          work, lwork,
                          info)
                end
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return A, tau, jpvt
        end
        # LQ with block reflectors
        #! format: off
        function gelqt!(A::AbstractMatrix{$elty},
                        T::AbstractMatrix{$elty}=similar(A, $elty, default_qr_blocksize(A), min(size(A)...)))
        #! format: on
            require_one_based_indexing(A, T)
            chkstride1(A)
            m, n = size(A)
            minmn = min(m, n)
            mb = size(T, 1)
            mb <= minmn ||
                throw(ArgumentError(lazy"block size $mb > $minmn too large"))
            size(T, 2) == minmn ||
                throw(DimensionMismatch(lazy"block reflector T should have size ($mb,$minmn)"))
            minmn == 0 && return A, T

            lda = max(1, stride(A, 2))
            ldt = max(1, stride(T, 2))
            work = Vector{$elty}(undef, mb * n)
            info = Ref{BlasInt}()
            ccall((@blasfunc($gelqt), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{BlasInt}),
                  m, n, mb, A, lda,
                  T, ldt, work,
                  info)
            chklapackerror(info[])
            return A, T
        end
        # LQ
        function gelqf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($gelqf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
        # QL
        function geqlf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($geqlf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
        # RQ
        function gerqf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            for i in 1:2                # first call returns lwork as work[1]
                ccall((@blasfunc($gerqf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = max(BlasInt(m), BlasInt(real(work[1])))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
    end
end

# Generate or multiply with Q factor

#! format: off
for (orglq, orgqr, orgql, orgrq, ormlq, ormqr, ormql, ormrq, gemqrt, elty) in
    ((:dorglq_, :dorgqr_, :dorgql_, :dorgrq_, :dormlq_, :dormqr_, :dormql_, :dormrq_, :dgemqrt_, :Float64),
     (:sorglq_, :sorgqr_, :sorgql_, :sorgrq_, :sormlq_, :sormqr_, :sormql_, :sormrq_, :sgemqrt_, :Float32),
     (:zunglq_, :zungqr_, :zungql_, :zungrq_, :zunmlq_, :zunmqr_, :zunmql_, :zunmrq_, :zgemqrt_, :ComplexF64),
     (:cunglq_, :cungqr_, :cungql_, :cungrq_, :cunmlq_, :cunmqr_, :cunmql_, :cunmrq_, :cgemqrt_, :ComplexF32))
#! format: on
    @eval begin
        # Multiply with blocked QR factor
        function gemqrt!(side::AbstractChar, trans::AbstractChar,
                         V::AbstractMatrix{$elty}, T::AbstractMatrix{$elty},
                         C::AbstractVecOrMat{$elty})
            require_one_based_indexing(V, T, C)
            chkstride1(V, T, C)
            chktrans(trans)
            chkside(side)
            m, n = size(C, 1), size(C, 2)
            nb, k = size(T)
            if k == 0
                return C
            end
            if side == 'L'
                if !(0 <= k <= m)
                    throw(DimensionMismatch(lazy"wrong value for k = $k: must be between 0 and $m"))
                end
                if m != size(V, 1)
                    throw(DimensionMismatch(lazy"first dimensions of C, $m, and V, $(size(V,1)) must match"))
                end
                wss = n * k
            elseif side == 'R'
                if !(0 <= k <= n)
                    throw(DimensionMismatch(lazy"wrong value for k = $k: must be between 0 and $n"))
                end
                if n != size(V, 1)
                    throw(DimensionMismatch(lazy"second dimension of C, $n, and first dimension of V, $(size(V,1)) must match"))
                end
                wss = m * k
            end
            if !(1 <= nb <= k)
                throw(DimensionMismatch(lazy"wrong value for nb = $nb, which must be between 1 and $k"))
            end
            ldv = max(1, stride(V, 2))
            ldt = max(1, stride(T, 2))
            ldc = max(1, stride(C, 2))
            work = Vector{$elty}(undef, wss)
            info = Ref{BlasInt}()
            ccall((@blasfunc($gemqrt), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8},
                   Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}, Clong, Clong),
                  side, trans,
                  m, n, k, nb,
                  V, ldv, T, ldt,
                  C, ldc,
                  work, info, 1, 1)
            chklapackerror(info[])
            return C
        end
        # Multiply with unblocked QR factor
        function ormqr!(side::AbstractChar, trans::AbstractChar,
                        A::AbstractMatrix{$elty}, tau::AbstractVector{$elty},
                        C::AbstractVecOrMat{$elty})
            require_one_based_indexing(A, tau, C)
            chkstride1(A, C, tau)
            chktrans(trans)
            chkside(side)
            m, n = size(C, 1), size(C, 2)
            mA = size(A, 1)
            k = length(tau)
            if side == 'L' && m != mA
                throw(DimensionMismatch(lazy"for a left-sided multiplication, the first dimension of C, $m, must equal the first dimension of A, $mA"))
            end
            if side == 'R' && n != mA
                throw(DimensionMismatch(lazy"for a right-sided multiplication, the second dimension of C, $n, must equal the first dimension of A, $mA"))
            end
            if side == 'L' && k > m
                throw(DimensionMismatch(lazy"invalid number of reflectors: k = $k should be <= m = $m"))
            end
            if side == 'R' && k > n
                throw(DimensionMismatch(lazy"invalid number of reflectors: k = $k should be <= n = $n"))
            end
            lda = max(1, stride(A, 2))
            ldc = max(1, stride(C, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($ormqr), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Clong, Clong),
                      side, trans,
                      m, n, k,
                      A, lda, tau,
                      C, ldc,
                      work, lwork, info, 1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return C
        end
        # Build Q factor in the space of `A`
        function orgqr!(A::AbstractMatrix{$elty}, tau::AbstractVector{$elty})
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            k = length(tau)
            m < n &&
                throw(DimensionMismatch(lazy"number of rows $m must be >= number of columns $n"))
            n < k &&
                throw(DimensionMismatch(lazy"tau has length $k, but needs (at most) length $n"))

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($orgqr), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, k,
                      A, lda, tau,
                      work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return A
        end
    end
end

# Symmetric / Hermitian eigenvalue decomposition
for (heev, heevx, heevr, heevd, hegvd, elty, relty) in
    ((:dsyev_, :dsyevx_, :dsyevr_, :dsyevd_, :dsygvd_, :Float64, :Float64),
     (:ssyev_, :ssyevx_, :ssyevr_, :ssyevd_, :ssygvd_, :Float32, :Float32),
     (:zheev_, :zheevx_, :zheevr_, :zheevd_, :zhegvd_, :ComplexF64, :Float64),
     (:cheev_, :cheevx_, :cheevr_, :cheevd_, :chegvd_, :ComplexF32, :Float32))
    @eval begin
        function heev!(A::AbstractMatrix{$elty},
                       W::AbstractVector{$relty}=similar(A, $relty, size(A, 1)),
                       V::AbstractMatrix{$elty}=A;
                       uplo::AbstractChar='U') # shouldn't matter but 'U' seems slightly faster than 'L'
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            if $elty <: Real
                issymmetric(A) || throw(ArgumentError("A must be symmetric"))
            else
                ishermitian(A) || throw(ArgumentError("A must be Hermitian"))
            end
            chkuplofinite(A, uplo)
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and V"))
                jobz = 'V'
            end

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if $elty <: Real
                for i in 1:2  # first call returns lwork as work[1]
                    ccall((@blasfunc($heev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W, work, lwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, max(1, 3n - 2))
                for i in 1:2  # first call returns lwork as work[1]
                    ccall((@blasfunc($heev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W, work, lwork, rwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end
            if jobz == 'V' && V !== A
                copy!(V, A)
            end
            return W, V
        end
        function heevx!(A::AbstractMatrix{$elty},
                        W::AbstractVector{$relty}=similar(A, $relty, size(A, 1)),
                        V::AbstractMatrix{$elty}=similar(A);
                        uplo::AbstractChar='U', # shouldn't matter but 'U' seems slightly faster than 'L'
                        kwargs...)
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            if $elty <: Real
                issymmetric(A) || throw(ArgumentError("A must be symmetric"))
            else
                ishermitian(A) || throw(ArgumentError("A must be Hermitian"))
            end
            chkuplofinite(A, uplo)
            if haskey(kwargs, :irange)
                il = first(irange)
                iu = last(irange)
                vl = vu = zero($relty)
                range = 'I'
            elseif haskey(kwargs, :vl) || haskey(kwargs, :vu)
                vl = convert($relty, get(kwargs, :vl, -Inf))
                vu = convert($relty, get(kwargs, :vu, +Inf))
                il = iu = 0
                range = 'V'
            else
                il = iu = 0
                vl = vu = zero($relty)
                range = 'A'
            end
            length(W) == n || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                jobz = 'V'
                size(V, 1) == n || throw(DimensionMismatch("size mismatch between A and V"))
                if range == 'I'
                    size(V, 2) >= iu - il + 1 ||
                        throw(DimensionMismatch("number of columns of V must correspond to number of requested eigenvalues"))
                else
                    size(V, 2) == n ||
                        throw(DimensionMismatch("size mismatch between A and V"))
                end
            end

            lda = max(1, stride(A, 2))
            ldv = max(1, stride(V, 2))
            abstol = -one($relty)
            m = Ref{BlasInt}()
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 5 * n)
            ifail = Vector{BlasInt}(undef, n)
            info = Ref{BlasInt}()
            if $elty <: Real
                for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$elty}, Ref{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                           Ptr{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobz, range, uplo,
                          n, A, lda,
                          vl, vu, il, iu, abstol, m,
                          W, V, ldv,
                          work, lwork, iwork, ifail,
                          info, 1, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, 7 * n)
                for i in 1:2  # first call returns lwork as work[1], lrwork as rwork[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$elty}, Ref{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                           Ptr{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                           Ptr{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobz, range, uplo,
                          n, A, lda,
                          vl, vu, il, iu, abstol, m,
                          W, V, ldv,
                          work, lwork, rwork, iwork, ifail,
                          info, 1, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end
            return W, V, m[]
        end
        function heevr!(A::AbstractMatrix{$elty},
                        W::AbstractVector{$relty}=similar(A, $relty, size(A, 1)),
                        V::AbstractMatrix{$elty}=similar(A);
                        uplo::AbstractChar='U', # shouldn't matter but 'U' seems slightly faster than 'L'
                        kwargs...)
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            if $elty <: Real
                issymmetric(A) || throw(ArgumentError("A must be symmetric"))
            else
                ishermitian(A) || throw(ArgumentError("A must be Hermitian"))
            end
            chkuplofinite(A, uplo)
            if haskey(kwargs, :irange)
                il = first(irange)
                iu = last(irange)
                vl = vu = zero($relty)
                range = 'I'
            elseif haskey(kwargs, :vl) || haskey(kwargs, :vu)
                vl = convert($relty, get(kwargs, :vl, -Inf))
                vu = convert($relty, get(kwargs, :vu, +Inf))
                il = iu = 0
                range = 'V'
            else
                il = iu = 0
                vl = vu = zero($relty)
                range = 'A'
            end
            length(W) == n || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                jobz = 'V'
                size(V, 1) == n || throw(DimensionMismatch("size mismatch between A and V"))
                if range == 'I'
                    size(V, 2) >= iu - il + 1 ||
                        throw(DimensionMismatch("number of columns of V must correspond to number of requested eigenvalues"))
                else
                    size(V, 2) == n ||
                        throw(DimensionMismatch("size mismatch between A and V"))
                end
            end

            lda = max(1, stride(A, 2))
            ldv = max(1, stride(V, 2))
            abstol = -one($relty)
            m = Ref{BlasInt}()
            isuppz = similar(A, BlasInt, 2 * n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if $elty <: Real
                for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevr), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$elty}, Ref{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                           Ptr{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobz, range, uplo,
                          n, A, lda,
                          vl, vu, il, iu, abstol, m,
                          W, V, ldv, isuppz,
                          work, lwork, iwork, liwork,
                          info, 1, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, 1)
                lrwork = BlasInt(-1)
                for i in 1:2  # first call returns lwork as work[1], lrwork as rwork[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevr), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$elty}, Ref{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                           Ptr{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobz, range, uplo,
                          n, A, lda,
                          vl, vu, il, iu, abstol, m,
                          W, V, ldv, isuppz,
                          work, lwork, rwork, lrwork, iwork, liwork,
                          info, 1, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        lrwork = BlasInt(rwork[1])
                        resize!(rwork, lrwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            end
            return W, V, m[]
        end
        function heevd!(A::AbstractMatrix{$elty},
                        W::AbstractVector{$relty}=similar(A, $relty, size(A, 1)),
                        V::AbstractMatrix{$elty}=A;
                        uplo::AbstractChar='U') # shouldn't matter but 'U' seems slightly faster than 'L'
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            if $elty <: Real
                issymmetric(A) || throw(ArgumentError("A must be symmetric"))
            else
                ishermitian(A) || throw(ArgumentError("A must be Hermitian"))
            end
            uplo = 'U' # shouldn't matter but 'U' seems slightly faster than 'L'
            chkuplofinite(A, uplo)
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and V"))
                jobz = 'V'
            end

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if $elty <: Real
                for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W, work, lwork, iwork, liwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, 1)
                lrwork = BlasInt(-1)
                for i in 1:2  # first call returns lwork as work[1], lrwork as rwork[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W,
                          work, lwork, rwork, lrwork, iwork, liwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        lrwork = BlasInt(rwork[1])
                        resize!(rwork, lrwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            end
            if jobz == 'V' && V !== A
                copy!(V, A)
            end
            return W, V
        end

        #         # Generalized eigenproblem
        #         #           SUBROUTINE DSYGVD( ITYPE, JOBZ, UPLO, N, A, LDA, B, LDB, W, WORK,
        #         #      $                   LWORK, IWORK, LIWORK, INFO )
        #         # *     .. Scalar Arguments ..
        #         #       CHARACTER          JOBZ, UPLO
        #         #       INTEGER            INFO, ITYPE, LDA, LDB, LIWORK, LWORK, N
        #         # *     ..
        #         # *     .. Array Arguments ..
        #         #       INTEGER            IWORK( * )
        #         #       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), W( * ), WORK( * )
        #         function sygvd!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar,
        #                         A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
        #             require_one_based_indexing(A, B)
        #             @chkvalidparam 1 itype 1:3
        #             @chkvalidparam 2 jobz ('N', 'V')
        #             chkuplo(uplo)
        #             chkstride1(A, B)
        #             n, m = checksquare(A, B)
        #             if n != m
        #                 throw(DimensionMismatch(lazy"dimensions of A, ($n,$n), and B, ($m,$m), must match"))
        #             end
        #             lda = max(1, stride(A, 2))
        #             ldb = max(1, stride(B, 2))
        #             w = similar(A, $elty, n)
        #             work = Vector{$elty}(undef, 1)
        #             lwork = BlasInt(-1)
        #             iwork = Vector{BlasInt}(undef, 1)
        #             liwork = BlasInt(-1)
        #             info = Ref{BlasInt}()
        #             for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
        #                 ccall((@blasfunc($sygvd), libblastrampoline), Cvoid,
        #                       (Ref{BlasInt}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
        #                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
        #                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
        #                        Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
        #                       itype, jobz, uplo, n,
        #                       A, lda, B, ldb,
        #                       w, work, lwork, iwork,
        #                       liwork, info, 1, 1)
        #                 chkargsok(info[])
        #                 if i == 1
        #                     lwork = BlasInt(work[1])
        #                     resize!(work, lwork)
        #                     liwork = iwork[1]
        #                     resize!(iwork, liwork)
        #                 end
        #             end
        #             chkposdef(info[])
        #             return w, A, B
        #         end
    end
end

# General eigenvalue decomposition
for (gees, geesx, geev, geevx, ggev, elty, celty, relty) in
    ((:sgees_, :sgeesx_, :sgeev_, :sgeevx_, :sggev_, :Float32, :ComplexF32, :Float32),
     (:dgees_, :dgeesx_, :dgeev_, :dgeevx_, :dggev_, :Float64, :ComplexF64, :Float64),
     (:cgees_, :cgeesx_, :cgeev_, :cgeevx_, :cggev_, :ComplexF32, :ComplexF32, :Float32),
     (:zgees_, :zgeesx_, :zgeev_, :zgeevx_, :zggev_, :ComplexF64, :ComplexF64, :Float64))
    @eval begin
        function gees!(A::AbstractMatrix{$elty},
                       V::AbstractMatrix{$elty}=similar(A),
                       vals::AbstractVector{$celty}=similar(A, $celty, size(A, 1)))
            require_one_based_indexing(A, V)
            chkstride1(A, V)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            if length(V) == 0
                jobvs = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and VR"))
                jobvs = 'V'
            end
            lda = max(1, stride(A, 2))
            ldv = max(1, stride(V, 2))
            sdim = Ref{BlasInt}()
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()

            if eltype(A) <: Real
                vals2 = reinterpret($elty, vals)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                valsR = view(vals2, 1:n)
                valsI = view(vals2, (n + 1):(2n))
                for i in 1:2  # first call returns lwork as work[1]
                    #! format: off
                    ccall((@blasfunc($gees), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvs, 'N', C_NULL,
                          n, A, lda, sdim,
                          valsR, valsI, V, ldv,
                          work, lwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, n)
                for i in 1:2
                    #! format: off
                    ccall((@blasfunc($gees), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvs, 'N', C_NULL,
                          n, A, lda, sdim,
                          vals, V, ldv,
                          work, lwork, rwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end

            # Cleanup the output in the real case
            if eltype(A) <: Real
                _reorder_realeigendecomposition!(vals, valsR, valsI, work, V, 'N')
            end
            return A, V, vals
        end
        function geev!(A::AbstractMatrix{$elty},
                       W::AbstractVector{$celty}=similar(A, $celty, size(A, 1)),
                       V::AbstractMatrix{$celty}=similar(A, $celty))
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobvr = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and VR"))
                jobvr = 'V'
            end
            jobvl = 'N'

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            VL = similar(A, n, 0)
            ldvl = max(1, stride(VL, 2))

            if eltype(A) <: Real
                W2 = reinterpret($elty, W)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                WR = view(W2, 1:n)
                WI = view(W2, (n + 1):(2n))
                VR = reinterpret($elty, V)
                ldvr = max(1, stride(VR, 2))
                for i in 1:2  # first call returns lwork as work[1]
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda,
                          WR, WI, VL, ldvl, VR, ldvr,
                          work, lwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                VR = V
                ldvr = max(1, stride(VR, 2))
                rwork = Vector{$relty}(undef, 2n)
                for i in 1:2
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda, 
                          W, VL, ldvl, VR, ldvr,
                          work, lwork, rwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end

            # Cleanup the output in the real case
            if eltype(A) <: Real
                _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
            end
            return W, V
        end
        function geevx!(A::AbstractMatrix{$elty},
                        W::AbstractVector{$celty}=similar(A, $celty, size(A, 1)),
                        V::AbstractMatrix{$celty}=similar(A, $celty);
                        scale::Bool=true, permute::Bool=true)
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobvr = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and VR"))
                jobvr = 'V'
            end
            jobvl = 'N'

            if scale && permute
                balanc = 'B'
            elseif scale
                balanc = 'S'
            elseif permute
                balanc = 'P'
            else
                balanc = 'N'
            end
            sense = 'N'

            lda = max(1, stride(A, 2))
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            scale = similar(A, $relty, n)
            abnrm = Ref{$relty}()
            rconde = similar(A, $relty, n)
            rcondv = similar(A, $relty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iworksize = 0
            iwork = Vector{BlasInt}(undef, iworksize)
            info = Ref{BlasInt}()
            VL = similar(A, n, 0)
            ldvl = max(1, stride(VL, 2))

            if eltype(A) <: Real
                W2 = reinterpret($elty, W)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                WR = view(W2, 1:n)
                WI = view(W2, (n + 1):(2n))
                VR = reinterpret($elty, V)
                ldvr = max(1, stride(VR, 2))
                for i in 1:2  # first call returns lwork as work[1]
                    #! format: off
                    ccall((@blasfunc($geevx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                           Ptr{$elty}, Ptr{$elty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                           Clong, Clong, Clong, Clong),
                          balanc, jobvl, jobvr, sense,
                          n, A, lda,
                          WR, WI, VL, ldvl, VR, ldvr,
                          ilo, ihi, scale, abnrm,
                          rconde, rcondv,
                          work, lwork, iwork, info,
                          1, 1, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                VR = V
                ldvr = max(1, stride(VR, 2))
                rwork = Vector{$relty}(undef, 2n)
                for i in 1:2
                    #! format: off
                    ccall((@blasfunc($geevx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                           Ptr{$relty}, Ptr{$relty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                           Clong, Clong, Clong, Clong),
                          balanc, jobvl, jobvr, sense,
                          n, A, lda,
                          W, VL, ldvl, VR, ldvr,
                          ilo, ihi, scale, abnrm,
                          rconde, rcondv,
                          work, lwork, rwork, info,
                          1, 1, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end

            # Cleanup the output in the real case
            if eltype(A) <: Real
                _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
            end
            return W, V
        end
        function ggev!(A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty},
                       W::AbstractVector{$celty}=similar(A, $celty, size(A, 1)),
                       V::AbstractMatrix{$celty}=similar(A, $celty))
            require_one_based_indexing(A, B, V, W)
            chkstride1(A, B, V, W)
            n = checksquare(A)
            n == checksquare(B) || throw(DimensionMismatch("size mismatch between A and B"))
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobvr = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and VR"))
                jobvr = 'V'
            end
            jobvl = 'N'

            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            VL = similar(A, n, 0)
            ldvl = stride(VL, 2)

            if eltype(A) <: Real
                W2 = reinterpret($elty, W)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                WR = view(W2, 1:n)
                WI = view(W2, (n + 1):(2n))
                VR = reinterpret($elty, V)
                ldvr = stride(VR, 2)
                for i in 1:2  # first call returns lwork as work[1]
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda,
                          WR, WI, VL, ldvl, VR, ldvr,
                          work, lwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                VR = V
                ldvr = stride(VR, 2)
                rwork = Vector{$relty}(undef, 2n)
                for i in 1:2
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda, 
                          W, VL, ldvl, VR, ldvr,
                          work, lwork, rwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end

            # Cleanup the output in the real case
            if eltype(A) <: Real
                _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
            end
            return W, V
        end
    end
end

function _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
    # first reorder eigenvalues and recycle work as temporary buffer to efficiently implement the permutation
    n = size(W, 1)
    resize!(work, n)
    copy!(work, WI)
    for i in n:-1:1
        W[i] = WR[i] + im * work[i]
    end
    if jobvr == 'V' # also reorganise vectors
        i = 1
        while i <= n
            if iszero(imag(W[i])) # real eigenvalue => real eigenvector
                for j in n:-1:1
                    VR[2 * j, i] = 0
                    VR[2 * j - 1, i] = VR[j, i]
                end
                i += 1
            else # complex eigenvalue => complex eigenvector and conjugate
                @assert i != n
                for j in n:-1:1
                    VR[2 * j, i] = VR[j, i + 1]
                    VR[2 * j - 1, i] = VR[j, i]
                    VR[2 * j, i + 1] = -VR[j, i + 1]
                    VR[2 * j - 1, i + 1] = VR[j, i]
                end
                i += 2
            end
        end
    end
end

# SVD
for (gesvd, gesdd, gesvdx, gejsv, gesvj, elty, relty) in
    ((:dgesvd_, :dgesdd_, :dgesvdx_, :dgejsv_, :dgesvj_, :Float64, :Float64),
     (:sgesvd_, :sgesdd_, :sgesvdx_, :sgejsv_, :sgesvj_, :Float32, :Float32),
     (:zgesvd_, :zgesdd_, :zgesvdx_, :zgejsv_, :zgesvj_, :ComplexF64, :Float64),
     (:cgesvd_, :cgesdd_, :cgesvdx_, :cgejsv_, :cgesvj_, :ComplexF32, :Float32))
    @eval begin
        #! format: off
        function gesvd!(A::AbstractMatrix{$elty},
                        S::AbstractVector{$relty}=similar(A, $relty, min(size(A)...)),
                        U::AbstractMatrix{$elty}=similar(A, $elty, size(A, 1), min(size(A)...)),
                        Vᴴ::AbstractMatrix{$elty}=similar(A, $elty, min(size(A)...), size(A, 2)))
        #! format: on
            require_one_based_indexing(A, U, Vᴴ, S)
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)
            if length(U) == 0
                jobu = 'N'
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        jobu = 'O'
                    else
                        jobu = 'S'
                    end
                elseif size(U, 2) == m
                    jobu = 'A'
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = 'N'
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        jobvt = 'O'
                    else
                        jobvt = 'S'
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = 'A'
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            if cmplx
                rwork = Vector{$relty}(undef, 5 * minmn)
            end
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                #! format: off
                if cmplx
                    ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobu, jobvt, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, rwork,
                          info, 1, 1)
                else
                    ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobu, jobvt, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork,
                          info, 1, 1)
                end
                #! format: on
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return (S, U, Vᴴ)
        end
        #! format: off
        function gesdd!(A::AbstractMatrix{$elty},
                        S::AbstractVector{$relty}=similar(A, $relty, min(size(A)...)),
                        U::AbstractMatrix{$elty}=similar(A, $elty, size(A, 1), min(size(A)...)),
                        Vᴴ::AbstractMatrix{$elty}=similar(A, $elty, min(size(A)...), size(A, 2)))
        #! format: on
            require_one_based_indexing(A, U, Vᴴ, S)
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)

            if length(U) == 0 && length(Vᴴ) == 0
                job = 'N'
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                length(S) == minmn ||
                    throw(DimensionMismatch("length mismatch between A and S"))
                if size(U, 2) == m && size(Vᴴ, 1) == n
                    job = 'A'
                elseif size(U, 2) == minmn && size(Vᴴ, 1) == minmn
                    if m >= n && U === A
                        job = 'O'
                    elseif m < n && Vᴴ === A
                        job = 'O'
                    else
                        job = 'S'
                    end
                else
                    throw(DimensionMismatch("invalid column size of U or row size of Vᴴ"))
                end
            end

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            if cmplx
                lrwork = job == 'N' ? 7 * minmn :
                         minmn * max(5 * minmn + 7, 2 * max(m, n) + 2 * minmn + 1)
                rwork = Vector{$relty}(undef, lrwork)
            end
            iwork = Vector{BlasInt}(undef, 8 * minmn)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                #! format: off
                if cmplx
                    ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                           Ptr{BlasInt}, Clong),
                          job, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, rwork, iwork,
                          info, 1)
                else
                    ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{BlasInt}, Clong),
                          job, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, iwork,
                          info, 1)
                end
                #! format: on
                chklapackerror(info[])
                if i == 1
                    # Work around issue with truncated Float32 representation of lwork in
                    # sgesdd by using nextfloat. See
                    # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
                    # and
                    # https://github.com/scipy/scipy/issues/5401
                    lwork = round(BlasInt, nextfloat(real(work[1])))
                    resize!(work, lwork)
                end
            end
            return (S, U, Vᴴ)
        end
        #! format: off
        function gesvdx!(A::AbstractMatrix{$elty},
                         S::AbstractVector{$relty}=similar(A, $relty, min(size(A)...)),
                         U::AbstractMatrix{$elty}=similar(A, $elty, size(A, 1), min(size(A)...)),
                         Vᴴ::AbstractMatrix{$elty}=similar(A, $elty, min(size(A)...), size(A, 2));
                         kwargs...)
        #! format: on
            require_one_based_indexing(A, U, Vᴴ, S)
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)
            if haskey(kwargs, :irange)
                il = first(irange)
                iu = last(irange)
                vl = vu = zero($relty)
                range = 'I'
            elseif haskey(kwargs, :vl) || haskey(kwargs, :vu)
                vl = convert($relty, get(kwargs, :vl, -Inf))
                vu = convert($relty, get(kwargs, :vu, +Inf))
                il = iu = 0
                range = 'V'
            else
                il = iu = 0
                vl = vu = zero($relty)
                range = 'A'
            end

            if length(U) == 0
                jobu = 'N'
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                size(U, 2) >= (range == 'I' ? iu - il + 1 : minmn) ||
                    throw(DimensionMismatch("invalid column size of U"))
                jobu = 'V'
            end
            if length(Vᴴ) == 0
                jobvt = 'N'
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                size(Vᴴ, 1) >= (range == 'I' ? iu - il + 1 : minmn) ||
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                jobvt = 'V'
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 12 * minmn)
            cmplx = eltype(A) <: Complex
            if cmplx
                rwork = Vector{$relty}(undef, minmn * (minmn * 2 + 15 * minmn)) # very strange specification in LAPACK docs: minmn * (minmn * 2 + 15 * minmn)
            end
            ns = Ref{BlasInt}()
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                #! format: off
                if cmplx
                    ccall((@blasfunc($gesvdx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$relty}, Ref{$relty}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobu, jobvt, range,
                          m, n, A, lda,
                          vl, vu, il, iu, ns,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, rwork, iwork,
                          info, 1, 1, 1)
                else
                    ccall((@blasfunc($gesvdx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$relty}, Ref{$relty}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobu, jobvt, range,
                          m, n, A, lda,
                          vl, vu, il, iu, ns,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, iwork,
                          info, 1, 1, 1)
                end
                #! format: on
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return (S, U, Vᴴ)
        end
        function gesvj!(A::AbstractMatrix{$elty},
                        S::AbstractVector{$relty}=similar(A, $relty, min(size(A)...)),
                        U::AbstractMatrix{$elty}=similar(A, $elty, size(A, 1),
                                                         min(size(A)...)),
                        Vᴴ::AbstractMatrix{$elty}=similar(A, $elty, min(size(A)...),
                                                          size(A, 2)))
            #! format: on
            require_one_based_indexing(A, U, Vᴴ, S)
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            m >= n ||
                throw(ArgumentError("gejsv! requires a matrix with at least as many rows as columns"))

            joba = 'G'
            if length(U) == 0
                jobu = 'N'
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == n
                    jobu = 'U'
                elseif size(U, 2) == m
                    throw(ArgumentError("Computation of full U matrix not supported in gesvj!"))
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobv = 'N'
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == n
                    jobv = 'V'
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == n ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            mv = Ref{BlasInt}() # unused
            if jobv == 'V'
                if U !== A
                    V = view(U, 1:n, 1:n) # use U as V storage
                else
                    V = view(similar(V), 1:n, 1:n)
                end
            else
                V = Vᴴ # doesn't matter, V is not used
            end
            ldv = max(1, stride(V, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            if cmplx
                rwork = Vector{$relty}(undef, 1)
                lrwork = BlasInt(-1)
            end
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1] and lrwork as rwork[1]
                #! format: off
                if cmplx
                    ccall((@blasfunc($gesvj), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$relty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          joba, jobu, jobv,
                          m, n, A, lda,
                          S, mv, V, ldv,
                          work, lwork, rwork, lrwork,
                          info, 1, 1, 1)
                else
                    ccall((@blasfunc($gesvj), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$relty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          joba, jobu, jobv,
                          m, n, A, lda,
                          S, mv, V, ldv,
                          work, lwork,
                          info, 1, 1, 1)
                end
                #! format: on
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    if cmplx
                        lrwork = BlasInt(real(rwork[1]))
                        resize!(rwork, lrwork)
                    end
                end
            end
            if jobv == 'V'
                adjoint!(Vᴴ, V)
            end
            if cmplx
                if !isone(rwork[1])
                    @warn "singular values might have underflowed or overflowed"
                    LinearAlgebra.rmul!(S, rwork[1])
                end
            else
                if !isone(work[1])
                    @warn "singular values might have underflowed or overflowed"
                    LinearAlgebra.rmul!(S, work[1])
                end
            end
            if jobu == 'U' && U !== A
                copyto!(U, A)
            end
            return (S, U, Vᴴ)
        end
    end
end

end
