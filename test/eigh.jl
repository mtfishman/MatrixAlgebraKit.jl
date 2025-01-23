@testset "eigh_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    for alg in (LAPACK_MultipleRelativelyRobustRepresentations(),
                LAPACK_DivideAndConquer(),
                LAPACK_QRIteration(),
                LAPACK_Bisection())
        A = randn(T, m, m)
        A = (A + A') / 2
        Ac = similar(A)
        V = similar(A)
        D = Diagonal(similar(A, real(T), m))
        Dc = similar(A, real(T), m)

        @constinferred eigh_full!(copy!(Ac, A), (D, V), alg)
        @test A * V ≈ V * Diagonal(D)
        @test V' * V ≈ I
        @test V * V' ≈ I
        @test all(isreal, D)

        D2, V2 = eigh_full!(copy!(Ac, A), (D, V), alg)
        @test D2 == D
        @test V2 == V

        @constinferred eigh_vals!(copy!(Ac, A), Dc, alg)
        @test D ≈ Diagonal(Dc)
    end
end

# @testset "eigh_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
#     m = 54
#     for alg in (MatrixAlgebraKit.RobustRepresentations(),
#                 LinearAlgebra.DivideAndConquer(),
#                 LinearAlgebra.QRIteration())
#         A = randn(T, m, m)
#         A = A * A'
#         A = (A + A') / 2
#         Ac = similar(A)
#         D₀ = reverse(eigh_vals!(copy!(Ac, A)))
#         r = m - 2
#         s = 1 + sqrt(eps(real(T)))

#         D, V = @constinferred eigh_trunc!(copy!(Ac, A); alg=alg, rank=r)
#         @test length(D) == r
#         @test LinearAlgebra.opnorm(A - V * Diagonal(D) * V') ≈ D₀[r + 1]

#         D, V = @constinferred eigh_trunc!(copy!(Ac, A); alg=alg,
#                                           atol=s * D₀[r + 1])
#         @test length(D) == r

#         D, V = @constinferred eigh_trunc!(copy!(Ac, A); alg=alg,
#                                           rtol=s * D₀[r + 1] / D₀[1])
#         @test length(D) == r

#         r1 = m - 6
#         r2 = m - 4
#         r3 = m - 2
#         D, V = @constinferred eigh_trunc!(copy!(Ac, A); alg=alg,
#                                           rank=r1,
#                                           atol=s * D₀[r2 + 1],
#                                           rtol=s * D₀[r3 + 1] / D₀[1])
#         @test length(D) == r1
#         D, V = @constinferred eigh_trunc!(copy!(Ac, A); alg=alg,
#                                           rank=r2,
#                                           atol=s * D₀[r3 + 1],
#                                           rtol=s * D₀[r1 + 1] / D₀[1])
#         @test length(D) == r1
#         D, V = @constinferred eigh_trunc!(copy!(Ac, A); alg=alg,
#                                           rank=r3,
#                                           atol=s * D₀[r1 + 1],
#                                           rtol=s * D₀[r2 + 1] / D₀[1])
#         @test length(D) == r1
#     end
# end
