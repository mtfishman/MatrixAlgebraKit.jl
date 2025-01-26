@testset "svd_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        k = min(m, n)
        @testset "algorithm $alg" for alg in
                                      (LAPACK_DivideAndConquer(), LAPACK_QRIteration(),
                                       LAPACK_Bisection(), LAPACK_Jacobi())
            n > m && alg isa LAPACK_Jacobi && continue # not supported
            A = randn(rng, T, m, n)

            U, S, Vᴴ = svd_compact(A, alg)
            @test size(U) == (m, k)
            @test size(S) == (k, k)
            @test size(Vᴴ) == (k, n)
            @test U * S * Vᴴ ≈ A
            @test U' * U ≈ I
            @test Vᴴ * Vᴴ' ≈ I
            @test isposdef(S)

            U2, S2, V2ᴴ = @constinferred svd_compact!(copy(A), (U, S, Vᴴ), alg)
            @test U2 == U
            @test S2 == S
            @test V2ᴴ == Vᴴ

            Sd = svd_vals(A, alg)
            @test S ≈ Diagonal(Sd)
        end
    end
end

@testset "svd_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        @testset "algorithm $alg" for alg in
                                      (LAPACK_DivideAndConquer(), LAPACK_QRIteration())
            A = randn(rng, T, m, n)

            U, S, Vᴴ = svd_full(A, alg)
            @test U * S * Vᴴ ≈ A
            @test U' * U ≈ I
            @test U * U' ≈ I
            @test Vᴴ * Vᴴ' ≈ I
            @test Vᴴ' * Vᴴ ≈ I
            @test all(isposdef, diagview(S))

            U2, S2, V2ᴴ = @constinferred svd_full!(copy(A), (U, S, Vᴴ), alg)
            @test U2 == U
            @test S2 == S
            @test V2ᴴ == Vᴴ
        end
    end
end

@testset "svd_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        @testset "algorithm $alg" for alg in
                                      (LAPACK_DivideAndConquer(), LAPACK_QRIteration(),
                                       LAPACK_Bisection(), LAPACK_Jacobi())
            n > m && alg isa LAPACK_Jacobi && continue # not supported
            A = randn(rng, T, m, n)
            S₀ = svd_vals(A)
            minmn = min(m, n)
            r = minmn - 2

            U1, S1, V1ᴴ = @constinferred svd_trunc(A; alg, trunc=truncrank(r))
            @test length(S1.diag) == r
            @test LinearAlgebra.opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]

            s = 1 + sqrt(eps(real(T)))
            trunc2 = trunctol(s * S₀[r + 1])
            alg2 = TruncatedAlgorithm(alg, trunc2)

            U2, S2, V2ᴴ = @constinferred svd_trunc(A; alg, trunc=trunctol(s * S₀[r + 1]))
            @test length(S2.diag) == r
            @test U1 ≈ U2
            @test S1 ≈ S2
            @test V1ᴴ ≈ V2ᴴ
        end
    end
end
