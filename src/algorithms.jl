"""
    abstract type AbstractAlgorithm end

Supertype to dispatch on specific implementations of different the different functions.
Concrete subtypes should represent both a way to dispatch to a given implementation, as
well as the configuration of that implementation.

See also [`select_algorithm`](@ref).
"""
abstract type AbstractAlgorithm end

"""
    Algorithm{name,KW} <: AbstractAlgorithm

Bare-bones implementation of an algorithm, where `name` should be a `Symbol` to dispatch on,
and `KW` is typically a `NamedTuple` indicating the keyword arguments.

See also [`@algdef`](@ref).
"""
struct Algorithm{name,K} <: AbstractAlgorithm
    kwargs::K
end

"""
    @algdef AlgorithmName

Convenience macro to define an algorithm `AlgorithmName` that accepts generic keywords.
This defines an exported alias for [`Algorithm{:AlgorithmName}`](@ref Algorithm)
along with some utility methods.
"""
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
