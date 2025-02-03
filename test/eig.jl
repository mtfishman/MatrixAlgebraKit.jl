@testset "eig_full! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(rng, T, m, m)
        Tc = complex(T)

        D, V = @constinferred eig_full(A; alg)
        @test eltype(D) == eltype(V) == Tc
        @test A * V ≈ V * D

        Ac = similar(A)
        D2, V2 = @constinferred eig_full!(copy!(Ac, A), (D, V), alg)
        @test D2 === D
        @test V2 === V
        @test A * V ≈ V * D

        Dc = @constinferred eig_vals(A, alg)
        @test eltype(Dc) == Tc
        @test D ≈ Diagonal(Dc)
    end
end

@testset "eig_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    rng = StableRNG(123)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(rng, T, m, m)
        A *= A' # TODO: deal with eigenvalue ordering etc
        # eigenvalues are sorted by ascending real component...
        D₀ = sort!(eig_vals(A); by=abs, rev=true)
        rmin = findfirst(i -> abs(D₀[end - i]) != abs(D₀[end - i - 1]), 1:(m - 2))
        r = length(D₀) - rmin
        trunc1 = truncrank(r)
        alg1 = TruncatedAlgorithm(alg, trunc1)

        D1, V1 = @constinferred eig_trunc(A, alg1)
        @test length(D1.diag) == r

        s = 1 + sqrt(eps(real(T)))
        alg2 = TruncatedAlgorithm(alg, trunctol(s * abs(D₀[r + 1])))
        D2, V2 = @constinferred eig_trunc(A, alg2)
        @test length(diagview(D2)) == r

        # # trunctol keeps order, truncrank does not
        # I1 = sortperm(diagview(D1); by=abs, rev=true)
        # I2 = sortperm(diagview(D2); by=abs, rev=true)
        # @test diagview(D1)[I1] ≈ diagview(D2)[I2]
        # @test V1[:, I1] ≈ V2[:, I2]
    end
end
