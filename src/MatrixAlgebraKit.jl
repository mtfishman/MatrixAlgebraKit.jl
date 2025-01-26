module MatrixAlgebraKit

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: Diagonal, diag, diagind
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, triu!

export qr_compact!, qr_full!
export svd_compact!, svd_full!, svd_vals!, svd_trunc!
export eigh_full!, eigh_vals!, eigh_trunc!
export eig_full!, eig_vals!, eig_trunc!
export truncrank, trunctol, TruncationKeepSorted, TruncationKeepFiltered

include("auxiliary.jl")
include("yalapack.jl")
include("interface/algorithms.jl")
include("interface/svd.jl")
include("decompositions/decompositions.jl")
include("decompositions/truncation.jl")
include("decompositions/qr.jl")
include("decompositions/svd.jl")
include("decompositions/eig.jl")
include("decompositions/eigh.jl")
include("decompositions/schur.jl")

end
