using Gen
using Distributions

MinProb() = 0.1

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
maybe_one_or_two_off(idx, prob, dom) = 
    (1 - prob) * onehot(idx, dom) +
    prob/3 * onehot(idx - 1, dom) +
    prob/3 * onehot(idx + 1, dom) +
    prob/6 * onehot(idx - 2, dom) +
    prob/6 * onehot(idx + 2, dom)

normalize(v) = v / sum(v)
discretized_gaussian(mean, std, dom) = normalize([
    cdf(Normal(mean, std), i + .5) - cdf(Normal(mean, std), i - .5) for i in dom
])
function truncate(pvec)
    mininvec = minimum(p for p in pvec if p != 0)
    if mininvec ≥ MinProb()
        return pvec
    else
        return truncate(normalize([p == mininvec ? 0. : p for p in pvec]))
    end
end
truncated_discretized_gaussian(args...) = discretized_gaussian(args...) |> truncate

truncate_value(val, range) = max(min(val, last(range)), first(range))

@dist labeled_categorical(labels, probs) = labels[categorical(probs)]

### Domains for discrete values ###
Positions() = 1:20; Vels() = -3:3;
Bools() = [true, false]

@gen (static) function step_model(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁)
    vxₜ ~ labeled_categorical(Vels(), maybe_one_off(vxₜ₋₁, 0.3, Vels()))
    vyₜ ~ labeled_categorical(Vels(), maybe_one_off(vyₜ₋₁, 0.3, Vels()))
    
    exp_x = xₜ₋₁ + vxₜ
    exp_y = yₜ₋₁ + vyₜ
    xₜ ~ categorical(maybe_one_off(exp_x, 0.6, Positions()))
    yₜ ~ categorical(maybe_one_off(exp_y, 0.6, Positions()))

    obsx ~ categorical(truncated_discretized_gaussian(xₜ, 2.0, Positions()))
    obsy ~ categorical(truncated_discretized_gaussian(yₜ, 2.0, Positions()))

    return obsx
end
@gen (static) function step_proposal(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁, obsx, obsy)
    projected_x = truncate_value(xₜ₋₁ + vxₜ₋₁, Positions())
    mean_x = (obsx + projected_x)/2
    xₜ ~ categorical(
        truncated_discretized_gaussian(mean_x, 1.5, Positions()) # TODO: make disc gauss support non-integer mean
    )
    diff_x = xₜ - xₜ₋₁
    vxₜ ~ labeled_categorical(Vels(),
        truncated_discretized_gaussian(diff_x, 1.0, Vels())
    )

    projected_y = truncate_value(yₜ₋₁ + vyₜ₋₁, Positions())
    mean_y = (obsx + projected_y)/2
    yₜ ~ categorical(
        truncated_discretized_gaussian(mean_y, 1.5, Positions()) # TODO: make disc gauss support non-integer mean
    )
    diff_y = yₜ - yₜ₋₁
    vyₜ ~ labeled_categorical(Vels(),
        truncated_discretized_gaussian(diff_y, 1.0, Vels())
    )
end

@load_generated_functions()