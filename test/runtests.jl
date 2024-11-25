using MatrixAlgebraKit
using Test
using TestExtras
using Aqua
using JET
using LinearAlgebra: LinearAlgebra, diag, Diagonal, I, isposdef, diagind

diagview(A) = view(A, diagind(A))

include("qr.jl")
include("svd.jl")
include("eigh.jl")

@testset "MatrixAlgebraKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MatrixAlgebraKit)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MatrixAlgebraKit; target_defined_modules=true)
    end
    # Write your tests here.
end
