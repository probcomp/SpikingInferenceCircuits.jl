using Distributions: Normal, cdf

VelCat = LCat(Vels())
BoolCat = LCat([true, false])

# positions(width) = positions the leftmost x position of an object with
# the given `width` could be placed in
positions(width) = 1:(ImageSideLength() - width + 1)

uniform(vals) = ones(length(vals))/length(vals)
	
onehot(x::Real, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 1. : 0. for i in dom]
onehot(x, dom) = [i == x ? 1. : 0. for i in dom]
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

_is_occluded(occ, x) = occ ≤ x ≤ occ + OccluderLength() - 1
_is_in_square(squarex, squarey, x, y) = (
    squarex ≤ x ≤ squarex + SquareSideLength() - 1 &&
    squarey ≤ y ≤ squarey + SquareSideLength() - 1
)

Map2D(gf) = Map(Map(gf))
Map2Dargs(args...) = collect(
    [
        arg[i, :]
        for i=1:size(arg)[1]
    ]
    for arg in args
)

truncate_value(val, dom) = 
    if val < minimum(dom)
        minimum(dom)
    elseif val > maximum(dom)
        maximum(dom)
    else
        val
    end

cond_normalize(vec) = sum(vec) > 0 ? normalize(vec) : normalize(ones(length(vec)))

pos_to_vel_dist(pos_t, pos_prev) = [
    let pos_this_would_get_us_to = truncate_value(pos_prev + v, positions(SquareSideLength()))
        pos_this_would_get_us_to == pos_t ? 1 : 0
    end
    for v in Vels()
        ] |> cond_normalize
