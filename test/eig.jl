@testset "eig_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(T, m, m)
        Ac = similar(A)
        Tc = complex(T)
        V = similar(A, Tc)
        D = Diagonal(similar(A, Tc, m))
        Dc = similar(A, Tc, m)

        @constinferred eig_full!(copy!(Ac, A), (D, V), alg)
        @test A * V ≈ V * D

        @constinferred eig_vals!(copy!(Ac, A), Dc, alg)
        @test D ≈ Diagonal(Dc)
    end
end