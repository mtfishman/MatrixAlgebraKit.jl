module MatrixAlgebraKit

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: Diagonal
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, triu!

export qr_compact!, qr_full!
export svd_compact!, svd_full!, svd_vals!, svd_trunc!
# export eigh_full!, eigh_vals!, eigh_trunc!
export truncrank, trunctol, TruncationKeepSorted, TruncationKeepFiltered

include("auxiliary.jl")
include("algorithms.jl")
include("truncation.jl")
include("yalapack.jl")
include("qr.jl")
include("svd.jl")
# include("eigh.jl")

end
