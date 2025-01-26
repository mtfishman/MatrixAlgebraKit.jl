@testset "qr_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        A = randn(rng, T, m, n)
        Ac = similar(A)
        Q = similar(A, m, min(m, n))
        R = similar(A, min(m, n), n)
        Q2 = similar(Q)
        noR = similar(A, min(m, n), 0)
        qr_compact!(copy!(Ac, A), (Q, R))
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_compact!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2
        # unblocked algorithm
        qr_compact!(copy!(Ac, A), (Q, R); blocksize=1)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_compact!(copy!(Ac, A), (Q2, noR); blocksize=1)
        @test Q == Q2
        if n <= m
            qr_compact!(copy!(Q2, A), (Q2, noR); blocksize=1) # in-place Q
            @test Q ≈ Q2
        end
        # other blocking
        qr_compact!(copy!(Ac, A), (Q, R); blocksize=8)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_compact!(copy!(Ac, A), (Q2, noR); blocksize=8)
        @test Q == Q2
        # pivoted
        qr_compact!(copy!(Ac, A), (Q, R); pivoted=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_compact!(copy!(Ac, A), (Q2, noR); pivoted=true)
        @test Q == Q2
        # positive
        qr_compact!(copy!(Ac, A), (Q, R); positive=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_compact!(copy!(Ac, A), (Q2, noR); positive=true)
        @test Q == Q2
        # positive and blocksize 1
        qr_compact!(copy!(Ac, A), (Q, R); positive=true, blocksize=1)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_compact!(copy!(Ac, A), (Q2, noR); positive=true, blocksize=1)
        @test Q == Q2
        # positive and pivoted
        qr_compact!(copy!(Ac, A), (Q, R); positive=true, pivoted=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        if n <= m
            # the following test tries to find the diagonal element (in order to test positivity)
            # before the column permutation. This only works if all columns have a diagonal
            # element
            for j in 1:n
                i = findlast(!iszero, view(R, :, j))
                @test real(R[i, j]) >= zero(real(T))
            end
        end
        qr_compact!(copy!(Ac, A), (Q2, noR); positive=true, pivoted=true)
        @test Q == Q2
    end
end

@testset "qr_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        A = randn(rng, T, m, n)
        Ac = similar(A)
        Q = similar(A, m, m)
        R = similar(A)
        Q2 = similar(Q)
        noR = similar(A, min(m, n), 0)
        qr_full!(copy!(Ac, A), (Q, R))
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_full!(copy!(Ac, A), (Q2, noR))
        @test Q == Q2
        # unblocked algorithm
        qr_full!(copy!(Ac, A), (Q, R); blocksize=1)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_full!(copy!(Ac, A), (Q2, noR); blocksize=1)
        @test Q == Q2
        if n == m
            qr_full!(copy!(Q2, A), (Q2, noR); blocksize=1) # in-place Q
            @test Q ≈ Q2
        end
        # other blocking
        qr_full!(copy!(Ac, A), (Q, R); blocksize=8)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_full!(copy!(Ac, A), (Q2, noR); blocksize=8)
        @test Q == Q2
        # pivoted
        qr_full!(copy!(Ac, A), (Q, R); pivoted=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        qr_full!(copy!(Ac, A), (Q2, noR); pivoted=true)
        @test Q == Q2
        # positive
        qr_full!(copy!(Ac, A), (Q, R); positive=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_full!(copy!(Ac, A), (Q2, noR); positive=true)
        @test Q == Q2
        # positive and blocksize 1
        qr_full!(copy!(Ac, A), (Q, R); positive=true, blocksize=1)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        @test all(>=(zero(real(T))), real(diag(R)))
        qr_full!(copy!(Ac, A), (Q2, noR); positive=true, blocksize=1)
        @test Q == Q2
        # positive and pivoted
        qr_full!(copy!(Ac, A), (Q, R); positive=true, pivoted=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        if n <= m
            # the following test tries to find the diagonal element (in order to test positivity)
            # before the column permutation. This only works if all columns have a diagonal
            # element
            for j in 1:n
                i = findlast(!iszero, view(R, :, j))
                @test real(R[i, j]) >= zero(real(T))
            end
        end
        qr_full!(copy!(Ac, A), (Q2, noR); positive=true, pivoted=true)
        @test Q == Q2
    end
end
