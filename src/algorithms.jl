struct Algorithm{name,K}
    kwargs::K
end

macro algdef(name)
    esc(quote
            const $name{K} = Algorithm{$(QuoteNode(name)),K}
            export $name
            function $name(; kwargs...)
                return $name{typeof(kwargs)}(kwargs)
            end
            function Base.print(io::IO, alg::$name)
                print(io, $name, "(")
                next = iterate(alg.kwargs)
                isnothing(next) && return print(io, ")")
                (k, v), state = next
                print(io, "; ", string(k), "=", string(v))
                next = iterate(alg.kwargs, state)
                while !isnothing(next)
                    (k, v), state = next
                    print(io, ", ", string(k), "=", string(v))
                    next = iterate(alg.kwargs, state)
                end
                return print(io, ")")
            end
        end)
end

# reference for naming LAPACK algorithms:
# https://www.netlib.org/lapack/explore-html/modules.html

# QR Decomposition
@algdef LAPACK_HouseholderQR

# General Eigenvalue Decomposition
@algdef LAPACK_Simple
@algdef LAPACK_Expert

const LAPACK_EigAlgorithm = Union{LAPACK_Simple,
                                  LAPACK_Expert}

# Hermitian Eigenvalue Decomposition
const LAPACK_QRIteration = LAPACK_Simple
export LAPACK_QRIteration
const LAPACK_Bisection = LAPACK_Expert
export LAPACK_Bisection
@algdef LAPACK_DivideAndConquer
@algdef LAPACK_MultipleRelativelyRobustRepresentations

const LAPACK_EighAlgorithm = Union{LAPACK_QRIteration,
                                   LAPACK_Bisection,
                                   LAPACK_DivideAndConquer,
                                   LAPACK_MultipleRelativelyRobustRepresentations}

# Singular Value Decomposition
@algdef LAPACK_Jacobi

const LAPACK_SVDAlgorithm = Union{LAPACK_QRIteration,
                                  LAPACK_Bisection,
                                  LAPACK_DivideAndConquer,
                                  LAPACK_Jacobi}
