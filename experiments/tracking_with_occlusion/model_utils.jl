LabeledCategorical(labels::Vector{T}, probs) where {T} = LabeledCPT{T}([[nothing]], labels, ((_,),) -> probs)
LabeledCategorical(labels, probs) = LabeledCategorical(collect(labels), probs)

# VelCat = LCat(Vels())
# BoolCat = LCat([true, false])

# positions(width) = positions the leftmost x position of an object with
# the given `width` could be placed in
positions(width) = 1:(ImageSideLength() - width)

uniform(vals) = ones(length(vals))/length(vals)
	
onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 1. : 0. for i in dom]
# prob vector to sample a value in `dom` which is 1 off
# from `idx` with probability `prob`, and `idx` otherwise
maybe_one_off(idx, prob, dom) =
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(idx - 1, dom) +
    prob/2 * onehot(idx + 1, dom)

discretized_gaussian(mean, std, dom) = normalize([
    cdf(Normal(mean, std), i + .5) - cdf(Normal(mean, std), i - .5) for i in dom
])

truncated_discretized_gaussian(mean, var, dom) =
    discretized_gaussian(mean, var, dom) |> truncate |> normalize

truncate(pvec) = [p ≥ MinProb() ? p : 0. for p in pvec]
function normalize(vec)
    @assert sum(vec) > 0.
    return vec/sum(vec)
end

Map2D(gf) = Map(Map(gf))
Map2Dargs(args...) = collect(
    [
        arg[i, :]
        for i=1:size(arg)[1]
    ]
    for arg in args
)