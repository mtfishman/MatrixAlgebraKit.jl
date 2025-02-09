@testset "left_null for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    n = m ÷ 2
    algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration())

    A = randn(rng, T, m, m)

    @testset "algorithm $alg" for alg in algs
        N = @constinferred left_null(A[:, 1:n]; alg)
        @test size(N) == (m - n, m)
        @test LinearAlgebra.norm(N * A[:, 1:n]) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test LinearAlgebra.rank([A[:, 1:n] N']) == m

        # test zeros
        Nz = @inferred left_null(zeros(T, m, m); alg)
        @test size(Nz) == (m, m)
        @test Nz ≈ LinearAlgebra.I
    end
end

@testset "right_null for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    n = m ÷ 2
    algs = (LAPACK_DivideAndConquer(), LAPACK_QRIteration())

    A = randn(rng, T, m, m)

    @testset "algorithm $alg" for alg in algs
        N = @constinferred right_null(A[1:n, :]; alg)
        @test size(N) == (m, m - n)
        @test LinearAlgebra.norm(A[1:n, :] * N) ≈ 0 atol = MatrixAlgebraKit.defaulttol(T)
        @test LinearAlgebra.rank([A[1:n, :]; N']) == m

        # test zeros
        Nz = @inferred right_null(zeros(T, m, m); alg)
        @test size(Nz) == (m, m)
        @test Nz ≈ LinearAlgebra.I
    end
end
