module MatrixAlgebraKit

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, triu!

export qr_compact!, qr_full!
export eigh_full!, eigh_vals!, eigh_trunc!
export svd_compact!, svd_full!, svd_vals!, svd_trunc!

include("auxiliary.jl")
include("backend.jl")
include("yalapack.jl")
include("qr.jl")
include("svd.jl")
include("eigh.jl")

end
