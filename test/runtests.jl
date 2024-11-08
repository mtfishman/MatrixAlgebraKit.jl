using MatrixAlgebraKit
using Test
using Aqua
using JET
using LinearAlgebra: diag, I

@testset "unsafe_qr for T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    # tall and square case
    m = 54
    for n in (37, m)
        A = randn(T, m, n)
        Ac = similar(A)
        Q = similar(A, m, n)
        R = similar(A, n, n)
        Q2 = similar(A, m, n)
        noR = similar(A, n, 0)
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q, R)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q2, noR)
        @test Q == Q2
        # unblocked algorithm
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q, R; blocksize=1)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q2, noR; blocksize=1)
        @test Q == Q2
        MatrixAlgebraKit._unsafe_qr!(copy!(Q2, A), Q2, noR; blocksize=1) # in-place Q
        @test Q ≈ Q2
        # other blocking
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q, R; blocksize=8)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q2, noR; blocksize=8)
        @test Q == Q2
        # pivoted
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q, R; pivoted=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q2, noR; pivoted=true)
        @test Q == Q2
        # positive
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q, R; positive=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        @test all(>=(zero(real(T))), real(diag(R)))
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q2, noR; positive=true)
        @test Q == Q2
        # positive and blocksize 1
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q, R; positive=true, blocksize=1)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        @test all(>=(zero(real(T))), real(diag(R)))
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q2, noR; positive=true, blocksize=1)
        @test Q == Q2
        # positive and pivoted
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q, R; positive=true, pivoted=true)
        @test Q * R ≈ A
        @test Q' * Q ≈ I
        for j in 1:n
            i = findlast(!iszero, view(R, :, j))
            @test real(R[i, j]) >= zero(real(T))
        end
        MatrixAlgebraKit._unsafe_qr!(copy!(Ac, A), Q2, noR; positive=true, pivoted=true)
        @test Q == Q2
    end
end

@testset "MatrixAlgebraKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MatrixAlgebraKit)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MatrixAlgebraKit; target_defined_modules=true)
    end
    # Write your tests here.
end
