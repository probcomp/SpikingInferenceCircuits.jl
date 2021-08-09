#=
This is a comparison between the fractional variance of the probability estimate for `p`, comparing:
1. Obtaining a probability estimate using simple-Monte-Carlo (ie. using K bernoulli(p) trials).
2. Obtaining a probability estimate by decomposing the outcome with probability `p` into a 2-variable
representation, where each variable has probability `sqrt(p)`, using simple Monte-Carlo with K/2 trials for
the estimate of each `sqrt(p)` probability, and then multiplying the estimates.
(We use K/2 rather than K because we spend half our time/space budget on each sample.)
This can be viewed as a pseudo-marginalization problem where we have a deterministic inverter from
the probability p outcome to the 2 probability sqrt(p) variable values.

This file both compares the analytic expressions I have derived for the fractional variance
using simple-Monte-Carlo and using the direct-inverter-for-two-sqrt(p)-outcomes to empirical
measurements of the variance, and compares the two methods against one another.
=#

using Distributions
using GLMakie

"""
Empirical estimate of the fractional variance of the estimate of `p` from simple Monte Carlo using 
`k` bernoulli samples.  Runs `nsamples` experiments for the empirical variance estimate.
"""
function direct_fractional_variance(p, k; nsamples=10000)
    Cs = [rand(Binomial(k, p)) for _=1:nsamples]
    pests = Cs / k
    fracvar = sum(((pest - p) / p)^2 for pest in pests) / nsamples
    return fracvar
end
"""
Empirical estimate of the fractional variance of the estimate of `p = sqrtp^2` from 2 simple monte carlo
estimates, each of which uses `halfk` bernoulli samples.  Runs `nsamples` experiments for the empirical variance estimate.
"""
function product_fractional_variance(sqrtp, halfk; nsamples=10000)
    pests = [
        rand(Binomial(halfk, sqrtp)) * rand(Binomial(halfk, sqrtp)) / halfk^2
        for _=1:nsamples
    ]
    fracvar = sum(
        ((pest - sqrtp^2) / sqrtp^2)^2 
        for pest in pests
    ) / nsamples
    return fracvar
end

"""Analytic expression for the fractional variance from simple Monte Carlo"""
analytic_direct_fracvar(p, k) = (1 - p)/(k * p)

"""Analytic expression for fractional variance from probability estimate by decomposing into 2 sqrt(p) outcomes."""
function analytic_prod_fracvar(sqrtp, halfk)
    k = 2 * halfk
    val = (1 + (2*(1-sqrtp))/(sqrtp * k))^2 - 1
    @assert val == (1 + analytic_direct_fracvar(sqrtp, halfk))^2 - 1
    return val
end


# Right now, all the plots take `k` as fixed, and vary `p` on the x axis

"""Compare simple Monte Carlo analytic variance expression to empirical value"""
function show_direct_works(f, ax)
    direct = lines!(ax, ps, [direct_fractional_variance(p, k) for p in ps])
    lines!(ax, ps, [analytic_direct_fracvar(p, k) for p in ps])
end
"""Compare decomposed representation analytic variance expression to empirical value"""
function show_prod_works(f, ax)
    prod = lines!(ax, ps, [product_fractional_variance(sqrt(p), k/2) for p in ps])
    lines!(ax, ps, [analytic_prod_fracvar(sqrt(p), k/2) for p in ps])
end

"""Compare 2 estimation schemes using empirical estimates"""
function show_comparison_empirical(f, ax)
    direct = lines!(ax, ps, [direct_fractional_variance(p, k) for p in ps])
    prod = lines!(ax, ps, [product_fractional_variance(sqrt(p), k/2) for p in ps])
    Legend(f[1, 2], [direct, prod], ["Simple Monte Carlo", "With Auxiliary Variable"])
end
"""Compare 2 estimation schemes using analytic expressions"""
function show_comparison_analytic(f, ax)
    direct = lines!(ax, ps, log.([analytic_direct_fracvar(p, k) for p in ps]))
    prod = lines!(ax, ps, log.([analytic_prod_fracvar(sqrt(p), k/2) for p in ps]))
    Legend(f[1, 2], [direct, prod], ["Simple Monte Carlo", "With Auxiliary Variable"])
end

"""Make a plot using the given `plttr` function"""
function mkplt(plttr)
    f = Figure()
    ax = Axis(f[1, 1], xlabel="p", ylabel="log fractional variance", title="Log Fractional Variance of probability estimate, for k = $(Int(k))")
    plttr(f, ax)
    f
end

# Code is set up so `k` is fixed to this global value:
latency = 5 # ms
frequency = 0.2 # spikes per ms
neuron_budget = 100
k = latency * neuron_budget * frequency

# Globally set `p` range to plot:
ps = 0.0001:0.0001:0.15

# [TODO: make `k`, `p` function arguments rather than globals]

mkplt(show_comparison_analytic)