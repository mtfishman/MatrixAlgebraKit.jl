using MatrixAlgebraKit
using Test
using TestExtras
using StableRNGs
using Aqua
using JET
using LinearAlgebra: LinearAlgebra, diag, Diagonal, I, isposdef, diagind

using MatrixAlgebraKit: diagview

@testset "QR Decomposition" verbose = true begin
    include("qr.jl")
end
@testset "Singular Value Decomposition" verbose = true begin
    include("svd.jl")
end
@testset "Hermitian Eigenvalue Decomposition" verbose = true begin
    include("eigh.jl")
end
@testset "General Eigenvalue Decomposition" verbose = true begin
    include("eig.jl")
end
@testset "Schur Decomposition" verbose = true begin
    include("schur.jl")
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
