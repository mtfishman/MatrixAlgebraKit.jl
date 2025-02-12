# User Interface

On the user-facing side of this package, we provide various implementations and interfaces for different matrix algebra operations.
These operations typically follow some common skeleton, and here we go into a little more detail to what behavior can be expected.

```@contents
Pages = ["user_interface.md"]
Depth = 2:3
```

## Compositions

Coming soon...

## Decompositions

A rather large class of matrix algebra methods consists of taking a single input `A`, and determining some factorization of that input.
In order to streamline these functions, they all follow a similar common code pattern.
For a given factorization `f`, this consists of the following methods:

```julia
f(A; kwargs...) -> F...
f!(A, [F]; kwargs...) -> F...
```

Here, the input matrix is always the first argument, and optionally the output can be provided as well.
The keywords are algorithm-specific, and can be used to influence the behavior of the algorithms.
Importantly, for generic code patterns it is recommended to always use the output `F` explicitly, since some implementations may not be able to reuse the provided memory.
Additionally, the `f!` method typically assumes that it is allowed to destroy the input `A`, and making use of the contents of `A` afterwards should be deemed as undefined behavior.

### QR Decomposition

The [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) transforms a matrix `A` into a product `Q * R`, where `Q` is orthonormal and `R` upper triangular.
This is often used to solve linear least squares problems, or construct orthogonal bases, since it is typically less expensive than the [Singular Value Decomposition](@ref).
If the input `A` is invertible, `Q` and `R` are unique if we require the diagonal elements of `R` to be positive.

For rectangular matrices `A` of size `(m, n)`, there are two modes of operation, [`qr_full`](@ref) and [`qr_compact`](@ref).
The former ensures that the resulting `Q` is a square unitary matrix of size `(m, m)`, while the latter creates an isometric `Q` of size `(m, min(m, n))`.

```@docs; canonical=false
qr_full
qr_compact
```

MatrixAlgebraKit ships with a LAPACK-based implementation for dense arrays, referred to as [`LAPACK_HouseholderQR`](@ref).
The additional configurations can be controlled with the following keyword arguments:

* `positive::Bool=false`: Ensure that the diagonal elements of `R` are non-negative.
* `pivoted::Bool=false`: Use [Column pivoting](https://en.wikipedia.org/wiki/QR_decomposition#Column_pivoting).
* `blocksize::Int`: Size of the blocks when using block-wise QR algorithm.

### Eigenvalue Decomposition

The [Eigenvalue Decomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) transforms a square matrix `A` into a product `V * D * V⁻¹`.
Equivalently, it finds `V` and `D` that satisfy `A * V = V * D`.

Not all matrices can be diagonalized, and some real matrices can only be diagonalized using complex arithmetic.
In particular, the resulting decomposition can only guaranteed to be real for real symmetric inputs `A`.
Therefore, we provide `eig_` and `eigh_` variants, where `eig` always results in complex-valued `V` and `D`, while `eigh` requires symmetric inputs but retains the scalartype of the input.

```@docs; canonical=false
eig_full
eig_trunc
eig_vals
eigh_full
eigh_trunc
eigh_vals
```

For the non-symmetric cases, there is [`LAPACK_Simple = LAPACK_QRIteration`](@ref LAPACK_QRIteration) and [`LAPACK_Expert = LAPACK_Bisection`](@ref LAPACK_Bisection).
For the symmetric cases, there additionally is [`LAPACK_DivideAndConquer`](@ref) and [`LAPACK_MultipleRelativelyRobustRepresentations`](@ref).

### Schur Decomposition

### Singular Value Decomposition

### Polar Decomposition

### Nullspaces

## Matrix functions

Coming soon...
