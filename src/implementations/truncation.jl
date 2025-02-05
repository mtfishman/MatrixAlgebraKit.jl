abstract type TruncationStrategy end

function TruncationStrategy(; atol=nothing, rtol=nothing, maxrank=nothing)
    if isnothing(maxrank) && isnothing(atol) && isnothing(rtol)
        return NoTruncation()
    elseif isnothing(maxrank)
        @assert isnothing(rtol) "TODO: rtol"
        return trunctol(atol)
    else
        return truncrank(maxrank)
    end
end

"""
    NoTruncation()

Trivial truncation strategy that keeps all values, mostly for testing purposes.
"""
struct NoTruncation <: TruncationStrategy end

"""
    TruncationKeepSorted(howmany::Int, sortby::Function, rev::Bool)

Truncation strategy to keep the first `howmany` values when sorted according to `sortby` or the last `howmany` if `rev` is true.
"""
struct TruncationKeepSorted{F} <: TruncationStrategy
    howmany::Int
    sortby::F
    rev::Bool
end

"""
    TruncationKeepFiltered(filter::Function)

Truncation strategy to keep the values for which `filter` returns true.
"""
struct TruncationKeepFiltered{F} <: TruncationStrategy
    filter::F
end

# TODO: better names for these functions of the above types
truncrank(howmany::Int, by=abs, rev=true) = TruncationKeepSorted(howmany, by, rev)
trunctol(atol) = TruncationKeepFiltered(≥(atol) ∘ abs)

# TODO: should we return a view?
function truncate!((U, S, Vᴴ)::Tuple{Vararg{AbstractMatrix,3}}, ind)
    return U[:, ind], Diagonal(diagview(S)[ind]), Vᴴ[ind, :]
end
function truncate!((D, V)::Tuple{Vararg{AbstractMatrix,2}}, ind)
    return Diagonal(diagview(D)[ind]), V[:, ind]
end

findtruncated(values::AbstractVector, ::NoTruncation) = trues(size(values))

# TODO: this may also permute the eigenvalues, decide if we want to allow this or not
# can be solved by going to simply sorting the resulting `ind`
function findtruncated(values::AbstractVector, strategy::TruncationKeepSorted)
    sorted = sortperm(values; by=strategy.sortby, rev=strategy.rev)
    howmany = min(strategy.howmany, length(sorted))
    ind = sorted[1:howmany]
    return ind # TODO: consider sort!(ind)
end

function findtruncated(values::AbstractVector, strategy::TruncationKeepFiltered)
    ind = findall(strategy.filter, values)
    return ind
end

"""
    TruncatedAlgorithm(alg::AbstractAlgorithm, trunc::TruncationAlgorithm)

Generic wrapper type for algorithms that consist of first using `alg`, followed by a
truncation through `trunc`.
"""
struct TruncatedAlgorithm{A,T} <: AbstractAlgorithm
    alg::A
    trunc::T
end
export TruncatedAlgorithm
