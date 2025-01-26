@testset "svd_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        @testset "algorithm $alg" for alg in
                                      (LAPACK_DivideAndConquer(), LAPACK_QRIteration(),
                                       LAPACK_Bisection(), LAPACK_Jacobi())
            n > m && alg isa LAPACK_Jacobi && continue # not supported
            A = randn(T, m, n)
            Ac = similar(A)
            U = similar(A, m, min(m, n))
            Vᴴ = similar(A, min(m, n), n)
            S = Diagonal(similar(A, real(T), min(m, n)))
            Sc = similar(A, real(T), min(m, n))

            svd_compact!(copy!(Ac, A), (U, S, Vᴴ), alg)
            @test U * S * Vᴴ ≈ A
            @test U' * U ≈ I
            @test Vᴴ * Vᴴ' ≈ I
            @test isposdef(S)

            U2, S2, V2ᴴ = @constinferred svd_compact!(copy!(Ac, A), (U, S, Vᴴ), alg)
            @test U2 == U
            @test S2 == S
            @test V2ᴴ == Vᴴ

            svd_vals!(copy!(Ac, A), Sc, alg)
            @test S ≈ Diagonal(Sc)
        end
    end
end

@testset "svd_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        @testset "algorithm $alg" for alg in
                                      (LAPACK_DivideAndConquer(), LAPACK_QRIteration())
            A = randn(T, m, n)
            Ac = similar(A)
            U = similar(A, m, m)
            Vᴴ = similar(A, n, n)
            S = similar(A, real(T), m, n)
            Sc = similar(S)

            svd_full!(copy!(Ac, A), (U, S, Vᴴ), alg)
            @test U * S * Vᴴ ≈ A
            @test U' * U ≈ I
            @test U * U' ≈ I
            @test Vᴴ * Vᴴ' ≈ I
            @test Vᴴ' * Vᴴ ≈ I
            @test all(isposdef, view(S, diagind(S)))

            U2, S2, V2ᴴ = @constinferred svd_full!(copy!(Ac, A), (U, S, Vᴴ), alg)
            @test U2 == U
            @test S2 == S
            @test V2ᴴ == Vᴴ
        end
    end
end

@testset "svd_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    @testset "size ($m, $n)" for n in (37, m, 63)
        @testset "algorithm $alg" for alg in
                                      (LAPACK_DivideAndConquer(), LAPACK_QRIteration(),
                                       LAPACK_Bisection(), LAPACK_Jacobi())
            n > m && alg isa LAPACK_Jacobi && continue # not supported
            A = randn(T, m, n)
            Ac = similar(A)
            S₀ = svd_vals!(copy!(Ac, A))
            minmn = min(m, n)
            r = minmn - 2
            trunc1 = truncrank(r)
            alg1 = TruncatedAlgorithm(alg, trunc1)

            U1, S1, V1ᴴ = @constinferred svd_trunc!(copy!(Ac, A), alg1)
            @test length(S1.diag) == r
            @test LinearAlgebra.opnorm(A - U1 * S1 * V1ᴴ) ≈ S₀[r + 1]

            s = 1 + sqrt(eps(real(T)))
            trunc2 = trunctol(s * S₀[r + 1])
            alg2 = TruncatedAlgorithm(alg, trunc2)

            U2, S2, V2ᴴ = @constinferred svd_trunc!(copy!(Ac, A), alg2)
            @test length(S2.diag) == r
            @test U1 ≈ U2
            @test S1 ≈ S2
            @test V1ᴴ ≈ V2ᴴ
        end
    end
end
