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

@algdef LAPACK_QRIteration
@algdef LAPACK_DivideAndConquer
@algdef LAPACK_RobustRepresentations
@algdef LAPACK_HouseholderQR
