module MatrixAlgebraKit

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, triu!

include("auxiliary.jl")
include("backend.jl")
include("yalapack.jl")
include("svd.jl")
include("eigh.jl")

end
