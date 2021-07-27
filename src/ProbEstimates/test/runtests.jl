using Gen
using ProbEstimates
using Test

normalize(pvec) = begin
    @assert sum(pvec) > 0
    pvec/sum(pvec)
end
onehot(v, vals) = [v == val ? 1. : 0. for val in vals]
unif(vals) = normalize([1. for _ in vals])
maybe_one_off(val, p, vals) = (
    (1 - p) * onehot(val, vals)     +
    p / 2   * onehot(val + 1, vals) +
    p / 2   * onehot(val - 1, vals)
) |> normalize

# TODO: tests for Cat, LCat

include("pseudomarginal.jl")