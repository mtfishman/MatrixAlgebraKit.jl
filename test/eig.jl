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

@testset "eig_trunc! for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    m = 54
    for alg in (LAPACK_Simple(), LAPACK_Expert())
        A = randn(T, m, m)
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
        trunc2 = trunctol(s * abs(D₀[r + 1]))
        alg2 = TruncatedAlgorithm(alg, trunc2)

        D2, V2 = @constinferred eig_trunc(A, alg2)
        @test length(D2.diag) == r

        # trunctol keeps order, truncrank does not
        I1 = sortperm(diagview(D1); by=abs, rev=true)
        I2 = sortperm(diagview(D2); by=abs, rev=true)
        @test diagview(D1)[I1] ≈ diagview(D2)[I2]
        @test V1[:, I1] ≈ V2[:, I2]
    end
end
