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
        return truncate(
            normalize([p == mininvec ? 0. : p for p in pvec])
        )
    end
end
truncated_discretized_gaussian(args...) = discretized_gaussian(args...) |> truncate
truncate_dist_to_valrange(pvec, range, dom) = [
    first(range) ≤ x ≤ last(range) ? p : 0.
    for (x, p) in zip(dom, pvec)
] |> normalize

truncate_value(val, range) = max(min(val, last(range)), first(range))

err_if_not_probvec(pvec, errmsg) =
    if isprobvec(pvec)
        pvec
    else
        error(errmsg)
    end

unif(range) = normalize([1. for _ in range])