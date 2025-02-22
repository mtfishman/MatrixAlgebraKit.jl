@testset "lq_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        L, Q = lq_compact(A)
        @test L isa Matrix{T} && size(L) == (m, minmn)
        @test Q isa Matrix{T} && size(Q) == (minmn, n)
        @test L * Q ≈ A
        @test Q * Q' ≈ I

        Ac = similar(A)
        L2, Q2 = @constinferred lq_compact!(copy!(Ac, A), (L, Q))
        @test L2 === L
        @test Q2 === Q

        noL = similar(A, 0, minmn)
        Q2 = similar(Q)
        lq_compact!(copy!(Ac, A), (noL, Q2))
        @test Q == Q2

        # unblocked algorithm
        lq_compact!(copy!(Ac, A), (L, Q); blocksize=1)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        lq_compact!(copy!(Ac, A), (noL, Q2); blocksize=1)
        @test Q == Q2
        if m <= n
            lq_compact!(copy!(Q2, A), (noL, Q2); blocksize=1) # in-place Q
            @test Q ≈ Q2
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (L, Q2); blocksize=1)
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (noL, Q2); positive=true)
            @test_throws ArgumentError lq_compact!(copy!(Q2, A), (noL, Q2); blocksize=8)
        end
        # other blocking: somehow blocksize=8 leads to strange memory errors
        lq_compact!(copy!(Ac, A), (L, Q); blocksize=16)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        lq_compact!(copy!(Ac, A), (noL, Q2); blocksize=16)
        @test Q == Q2
        # pivoted
        @test_throws ArgumentError lq_compact!(copy!(Ac, A), (L, Q); pivoted=true)
        # positive
        lq_compact!(copy!(Ac, A), (L, Q); positive=true)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_compact!(copy!(Ac, A), (noL, Q2); positive=true)
        @test Q == Q2
        # positive and blocksize 1
        lq_compact!(copy!(Ac, A), (L, Q); positive=true, blocksize=1)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_compact!(copy!(Ac, A), (noL, Q2); positive=true, blocksize=1)
        @test Q == Q2
    end
end

@testset "lq_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for n in (37, m, 63)
        minmn = min(m, n)
        A = randn(rng, T, m, n)
        L, Q = lq_full(A)
        @test L isa Matrix{T} && size(L) == (m, n)
        @test Q isa Matrix{T} && size(Q) == (n, n)
        @test L * Q ≈ A
        @test Q * Q' ≈ I

        Ac = similar(A)
        L2, Q2 = @constinferred lq_full!(copy!(Ac, A), (L, Q))
        @test L2 === L
        @test Q2 === Q
        @test L * Q ≈ A
        @test Q * Q' ≈ I

        noL = similar(A, 0, n)
        Q2 = similar(Q)
        lq_full!(copy!(Ac, A), (noL, Q2))
        @test Q == Q2

        # unblocked algorithm
        lq_full!(copy!(Ac, A), (L, Q); blocksize=1)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        lq_full!(copy!(Ac, A), (noL, Q2); blocksize=1)
        @test Q == Q2
        if n == m
            lq_full!(copy!(Q2, A), (noL, Q2); blocksize=1) # in-place Q
            @test Q ≈ Q2
        end
        # # other blocking
        lq_full!(copy!(Ac, A), (L, Q); blocksize=16)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        lq_full!(copy!(Ac, A), (noL, Q2); blocksize=16)
        @test Q == Q2
        # pivoted
        @test_throws ArgumentError lq_full!(copy!(Ac, A), (L, Q); pivoted=true)
        # positive
        lq_full!(copy!(Ac, A), (L, Q); positive=true)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2); positive=true)
        @test Q == Q2
        # positive and blocksize 1
        lq_full!(copy!(Ac, A), (L, Q); positive=true, blocksize=1)
        @test L * Q ≈ A
        @test Q * Q' ≈ I
        @test all(>=(zero(real(T))), real(diag(L)))
        lq_full!(copy!(Ac, A), (noL, Q2); positive=true, blocksize=1)
        @test Q == Q2
    end
end
