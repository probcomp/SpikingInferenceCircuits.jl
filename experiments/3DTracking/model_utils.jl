onehot(x, dom) =
    !isfinite(x) ? unif(dom) : 
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
    [i == x ? 1. : 0. for i in dom] 
# prob vector to sample a value in `dom` which is 1 off
# from `idx` with probability `prob`, and `idx` otherwise
maybe_one_off(idx, prob, dom) =
    !isfinite(idx) ? unif(dom) :
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(round(idx - abs(dom[2]-dom[1]), digits=1), dom) +
    prob/2 * onehot(round(idx + abs(dom[2]-dom[1]), digits=1), dom)
maybe_one_or_two_off(idx, prob, dom) =
    !isfinite(idx) ? unif(dom) : 
    (1 - prob) * onehot(idx, dom) +
    prob/3 * onehot(round(idx - abs(dom[2]-dom[1]), digits=1), dom) +
    prob/3 * onehot(round(idx + abs(dom[2]-dom[1]), digits=1), dom) +
    prob/6 * onehot(round(idx - 2*abs(dom[2]-dom[1]), digits=1), dom) +
    prob/6 * onehot(round(idx + 2*abs(dom[2]-dom[1]), digits=1), dom)


# function onehot(x, dom)
#     @assert(isfinite(x))
#     !isfinite(x) ? unif(dom) : 
#     x < first(dom) ? onehot(first(dom), dom) :
#     x > last(dom)  ? onehot(last(dom), dom)  :
#     [i == x ? 1. : 0. for i in dom]
# end

# # prob vector to sample a value in `dom` which is 1 off
# # from `idx` with probability `prob`, and `idx` otherwise
# function maybe_one_off(idx, prob, dom)
#     @assert(isfinite(idx))
#     !isfinite(idx) ? unif(dom) :
#     (1 - prob) * onehot(idx, dom) +
#     prob/2 * onehot(round(idx - abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/2 * onehot(round(idx + abs(dom[2]-dom[1]), digits=1), dom)
# end

# function maybe_one_or_two_off(idx, prob, dom)
#     @assert(isfinite(idx))
#     !isfinite(idx) ? unif(dom) : 
#     (1 - prob) * onehot(idx, dom) +
#     prob/3 * onehot(round(idx - abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/3 * onehot(round(idx + abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/6 * onehot(round(idx - 2*abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/6 * onehot(round(idx + 2*abs(dom[2]-dom[1]), digits=1), dom)
# end






# onehot(x, dom) =
#     x < first(dom) ? onehot(first(dom), dom) :
#     x > last(dom)  ? onehot(last(dom), dom)  :
#     [i == x ? 1. : 0. for i in dom] 
# # prob vector to sample a value in `dom` which is 1 off
# # from `idx` with probability `prob`, and `idx` otherwise
# maybe_one_off(idx, prob, dom) =
#     (1 - prob) * onehot(idx, dom) +
#     prob/2 * onehot(round(idx - abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/2 * onehot(round(idx + abs(dom[2]-dom[1]), digits=1), dom)
# maybe_one_or_two_off(idx, prob, dom) =
#     (1 - prob) * onehot(idx, dom) +
#     prob/3 * onehot(round(idx - abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/3 * onehot(round(idx + abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/6 * onehot(round(idx - 2*abs(dom[2]-dom[1]), digits=1), dom) +
#     prob/6 * onehot(round(idx + 2*abs(dom[2]-dom[1]), digits=1), dom)




normalize(v) = v / sum(v) 
#normalize(v) = sum(v) > 0 ? v / sum(v) : [1/length(v) for i in v]

function discretized_gaussian(mean, std, dom)
    if mean > dom[end]
        retvec = zeros(length(dom))
        retvec[end] = 1.0
        return retvec
    elseif mean < dom[1]
        retvec = zeros(length(dom))
        retvec[1] = 1.0
        return retvec
    else
        return normalize([
            cdf(Distributions.Normal(mean, std), i + abs(dom[2]-dom[1])/2) - cdf(Distributions.Normal(mean, std), i - abs(dom[2]-dom[1])/2) for i in dom
                ])
    end
end


function truncate(pvec)
    if !isprobvec(pvec)
        error("pvec = $pvec is not a probability vector")
    end
    mininvec = minimum(p for p in pvec if p != 0)
    if mininvec ≥ MinProb()
        return pvec
    else
        first_to_truncate = findfirst(pvec .== mininvec)
        return truncate(
            normalize([i == first_to_truncate ? 0. : p for (i, p) in enumerate(pvec)])
        )
    end
end

truncated_discretized_gaussian(args...) = discretized_gaussian(args...) |> truncate
truncate_dist_to_valrange(pvec, range, dom) = [
    first(range) ≤ x ≤ last(range) ? p : 0.
    for (x, p) in zip(dom, pvec)
] |> normalize

truncate_value(val, range) = max(min(val, last(range)), first(range))

unif(range) = normalize([1. for _ in range])
