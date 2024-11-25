@testset "svd_compact! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    for n in (37, m, 63)
        for alg in (LinearAlgebra.DivideAndConquer(), LinearAlgebra.QRIteration())
            A = randn(T, m, n)
            Ac = similar(A)
            U = similar(A, m, min(m, n))
            Vᴴ = similar(A, min(m, n), n)
            S = similar(A, real(T), min(m, n))
            Sc = similar(S)

            svd_compact!(copy!(Ac, A), U, S, Vᴴ; alg=alg)
            @test U * Diagonal(S) * Vᴴ ≈ A
            @test U' * U ≈ I
            @test Vᴴ * Vᴴ' ≈ I
            @test all(isposdef, S)

            U2, S2, V2ᴴ = @constinferred svd_compact!(copy!(Ac, A), U, S, Vᴴ; alg=alg)
            @test U2 == U
            @test S2 == S
            @test V2ᴴ == Vᴴ

            svd_vals!(copy!(Ac, A), Sc; alg=alg)
            @test S ≈ Sc

            S3 = @constinferred svd_vals!(copy!(Ac, A); alg=alg)
            @test S3 == Sc
        end
    end
end

@testset "svd_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    for n in (37, m, 63)
        for alg in (LinearAlgebra.DivideAndConquer(), LinearAlgebra.QRIteration())
            A = randn(T, m, n)
            Ac = similar(A)
            U = similar(A, m, m)
            Vᴴ = similar(A, n, n)
            S = similar(A, real(T), min(m, n))
            Sc = similar(S)
            Σ = zero(A)

            svd_full!(copy!(Ac, A), U, S, Vᴴ; alg=alg)
            copy!(diagview(Σ), S)
            @test U * Σ * Vᴴ ≈ A
            @test U' * U ≈ I
            @test U * U' ≈ I
            @test Vᴴ * Vᴴ' ≈ I
            @test Vᴴ' * Vᴴ ≈ I
            @test all(isposdef, S)

            U2, S2, V2ᴴ = @constinferred svd_full!(copy!(Ac, A), U, S, Vᴴ; alg=alg)
            @test U2 == U
            @test S2 == S
            @test V2ᴴ == Vᴴ

            svd_vals!(copy!(Ac, A), Sc; alg=alg)
            @test S ≈ Sc

            S3 = @constinferred svd_vals!(copy!(Ac, A); alg=alg)
            @test S3 == Sc
        end
    end
end

@testset "svd_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    for n in (37, m, 63)
        for alg in (LinearAlgebra.DivideAndConquer(), LinearAlgebra.QRIteration())
            A = randn(T, m, n)
            Ac = similar(A)
            U = similar(A, m, m)
            Vᴴ = similar(A, n, n)
            S = similar(A, real(T), min(m, n))
            Sc = similar(S)
            Σ = zero(A)

            svd_full!(copy!(Ac, A), U, S, Vᴴ; alg=alg)
            copy!(diagview(Σ), S)
            @test U * Σ * Vᴴ ≈ A
            @test U' * U ≈ I
            @test U * U' ≈ I
            @test Vᴴ * Vᴴ' ≈ I
            @test Vᴴ' * Vᴴ ≈ I
            @test all(isposdef, S)

            U2, S2, V2ᴴ = @constinferred svd_full!(copy!(Ac, A), U, S, Vᴴ; alg=alg)
            @test U2 == U
            @test S2 == S
            @test V2ᴴ == Vᴴ

            svd_vals!(copy!(Ac, A), Sc; alg=alg)
            @test S ≈ Sc

            S3 = @constinferred svd_vals!(copy!(Ac, A); alg=alg)
            @test S3 == Sc
        end
    end
end

@testset "svd_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    for n in (37, m, 63)
        for alg in (LinearAlgebra.DivideAndConquer(), LinearAlgebra.QRIteration())
            A = randn(T, m, n)
            Ac = similar(A)
            S₀ = svd_vals!(copy!(Ac, A))
            minmn = min(m, n)
            r = minmn - 2
            s = 1 + sqrt(eps(real(T)))

            U, S, Vᴴ = @constinferred svd_trunc!(copy!(Ac, A); alg=alg, rank=r)
            @test length(S) == r
            @test LinearAlgebra.opnorm(A - U * Diagonal(S) * Vᴴ) ≈ S₀[r + 1]

            U, S, Vᴴ = @constinferred svd_trunc!(copy!(Ac, A); alg=alg,
                                                 atol=s * S₀[r + 1])
            @test length(S) == r

            U, S, Vᴴ = @constinferred svd_trunc!(copy!(Ac, A); alg=alg,
                                                 rtol=s * S₀[r + 1] / S₀[1])
            @test length(S) == r

            r1 = minmn - 6
            r2 = minmn - 4
            r3 = minmn - 2
            U, S, Vᴴ = @constinferred svd_trunc!(copy!(Ac, A); alg=alg,
                                                 rank=r1,
                                                 atol=s * S₀[r2 + 1],
                                                 rtol=s * S₀[r3 + 1] / S₀[1])
            @test length(S) == r1
            U, S, Vᴴ = @constinferred svd_trunc!(copy!(Ac, A); alg=alg,
                                                 rank=r2,
                                                 atol=s * S₀[r3 + 1],
                                                 rtol=s * S₀[r1 + 1] / S₀[1])
            @test length(S) == r1
            U, S, Vᴴ = @constinferred svd_trunc!(copy!(Ac, A); alg=alg,
                                                 rank=r3,
                                                 atol=s * S₀[r1 + 1],
                                                 rtol=s * S₀[r2 + 1] / S₀[1])
            @test length(S) == r1
        end
    end
end
