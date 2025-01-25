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
name(alg::Algorithm) = name(typeof(alg))
name(::Type{<:Algorithm{N}}) where {N} = N

# TODO: do we want to restrict this to Algorithm{name,<:NamedTuple}?
# Pretend like kwargs are part of the properties of the algorithm
Base.propertynames(alg::Algorithm) = (:kwargs, propertynames(getfield(alg, :kwargs))...)
@inline function Base.getproperty(alg::Algorithm, f::Symbol)
    kwargs = getfield(alg, :kwargs)
    return f === :kwargs ? kwargs : getproperty(kwargs, f)
end

# TODO: do we want to simply define this for all `Algorithm{N,<:NamedTuple}`?
# need print to make strings/symbols parseable,
# show to make objects parseable
function _show_alg(io::IO, alg::Algorithm)
    print(io, name(alg))
    print(io, "(")
    properties = propertynames(alg)
    next = iterate(properties)
    isnothing(next) && return print(io, ")")
    f, state = next
    print(io, "; ", f, "=")
    show(io, getproperty(alg, f))
    next = iterate(properties, state)
    while !isnothing(next)
        f, state = next
        print(io, ", ", f, "=")
        show(io, getproperty(alg, f))
        next = iterate(properties, state)
    end
    return print(io, ")")
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
                # TODO: is this necessary/useful?
                kw = NamedTuple(kwargs) # normalize type
                return $name{typeof(kw)}(kw)
            end
            function Base.show(io::IO, alg::$name)
                return _show_alg(io, alg)
            end
        end)
end
