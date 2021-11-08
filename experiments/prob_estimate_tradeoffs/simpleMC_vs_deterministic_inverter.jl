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
function show_logcomparison_empirical(f, ax)
    direct = lines!(ax, ps, [direct_fractional_variance(p, k) for p in ps])
    prod = lines!(ax, ps, [product_fractional_variance(sqrt(p), k/2) for p in ps])
    Legend(f[1, 2], [direct, prod], ["Simple Monte Carlo", "With Auxiliary Variable"])
end
"""Compare 2 estimation schemes using analytic expressions"""
function show_logcomparison_analytic(f, ax)
    direct = lines!(ax, ps, log.([analytic_direct_fracvar(p, k) for p in ps]))
    prod = lines!(ax, ps, log.([analytic_prod_fracvar(sqrt(p), k/2) for p in ps]))
    Legend(f[1, 2], [direct, prod], ["Simple Monte Carlo", "With Auxiliary Variable"])
end
function show_directcomparison_analytic(f, ax)
    direct = lines!(ax, ps, ([analytic_direct_fracvar(p, k) for p in ps]))
    prod = lines!(ax, ps, ([analytic_prod_fracvar(sqrt(p), k/2) for p in ps]))
    Legend(f[1, 2], [direct, prod], ["Without Auxiliary Variables", "With Auxiliary Variable"])
end

"""Make a plot using the given `plttr` function"""
function mkplt_singlevar(plttr, is_log=(plttr in (show_logcomparison_analytic, show_logcomparison_empirical)))
    f = Figure()
    logstring = is_log ? "Log " : ""
    ax = Axis(f[1, 1], xlabel="p", ylabel="$(logstring)fractional variance", title="$(logstring)Fractional Variance of probability estimate, for k = $(Int(k))")
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

# mkplt_singlevar(show_logcomparison_analytic)

### Distributed probability estimate ###
"""
- p_values: array of probability values
- k_per_estimate: `k` value (number of spikes to accumulate) used to estimate each of the probability values
- single_estimate_fracvar : function from (p, num_neurons) to fractional variance of estimating that value
"""
function analytic_logfracvar_for_multiestimate_product(p_values, k_per_estimate, single_estimate_fracvar)
    fracvars = [
        single_estimate_fracvar(p_value, k_per_estimate)
        for p_value in p_values
    ]
    product_fracvar = prod((fv + 1.) for fv in fracvars) - 1.
    return log(product_fracvar)
end

function analytic_logfracvar_for_num_pixelflips(p_flip, n_flip, n_pixels, k_per_estimate, single_estimate_fracvar)
    analytic_logfracvar_for_multiestimate_product(
        [i ≤ n_flip ? p_flip : (1 - p_flip) for i=1:n_pixels],
        k_per_estimate, single_estimate_fracvar
    )
end
analytic_logfracvar_for_num_pixelflips_noaux(args...) = analytic_logfracvar_for_num_pixelflips(args..., analytic_direct_fracvar)
analytic_logfracvar_for_num_pixelflips_withaux(args...) = analytic_logfracvar_for_num_pixelflips(args..., (p, k) -> product_fractional_variance(√(p), k/2))

function mkplt_distributed(sidewidth, p_pix_flip, n_flips_range)
    noaux_vals = exp.([analytic_logfracvar_for_num_pixelflips_noaux(p_pix_flip, n_flip, sidewidth^2, k) for n_flip in n_flips_range])
    aux_vals = exp.([analytic_logfracvar_for_num_pixelflips_withaux(p_pix_flip, n_flip, sidewidth^2, k) for n_flip in n_flips_range])

    f = Figure()
    ax = Axis(f[1, 1],
        xlabel="Number of Pixel Flips",
        ylabel="Fractional variance",
        title="Fractional Variance of Image Likelihood. $(sidewidth)x$(sidewidth) image. P_pix_flip=$p_pix_flip. $neuron_budget neurons/pixel; E[latency]=$latency; max neuron frequency=$frequency KHz. "
    )

    noaux = lines!(ax, n_flips_range, noaux_vals)
    withaux = lines!(ax, n_flips_range, aux_vals)
    Legend(f[1, 2], [noaux, withaux], ["Without Auxiliary Variables", "With Auxiliary Variables"])

    f
end

# mkplt_distributed(10, 0.01, 1:5)

### Single-pixel latency at fixed fractional variance
function noaux_k_for_fracvar(p, fracvar)
    k = (1 - p)/(p * fracvar)
    @assert isapprox(analytic_direct_fracvar(p, k), fracvar)
    @assert round(k) ≥ 1
    round(k)
end
function withaux_k_for_fracvar(p, fracvar)
    # k = (1 + 2*(1 - √(p))) / √(p * (1 + fracvar))
    f = fracvar
    k = 2*(√((f + 1) * (√(p) - p)^2) - p + √(p)) / (f * p)
    @assert round(k) ≥ 1
    @assert isapprox(analytic_prod_fracvar(√(p), k/2), fracvar) "$(analytic_prod_fracvar(√(p), k/2)) | $fracvar"
    round(k)
end
function k_to_Elatency(k, num_neurons, frequency)
    Elatency = k / (num_neurons * frequency)
    @assert isapprox(k, Elatency * num_neurons * frequency) "$k | $(Elatency * num_neurons * frequency)"
    Elatency
end
function latency_plot(;fracvar=0.5, probs=ps, num_neurons=neuron_budget, frequency=frequency)
    noaux_latencies = [k_to_Elatency(noaux_k_for_fracvar(p, fracvar), num_neurons, frequency) for p in probs]
    withaux_latencies = [k_to_Elatency(withaux_k_for_fracvar(p, fracvar), num_neurons, frequency) for p in probs]

    f = Figure()
    ax = Axis(f[1, 1],
        xlabel="Pixel Flip Probability",
        ylabel="Expected Latency",
        title="Expected latency (ms) to score pixel flip at fractional variance $fracvar using $neuron_budget neurons, each at $frequency KHz. "
    )

    ylims!(ax, (0, 100))

    noaux = lines!(ax, probs, noaux_latencies)
    withaux = lines!(ax, probs, withaux_latencies)
    Legend(f[1, 2], [noaux, withaux], ["Without Auxiliary Variables", "With Auxiliary Variables"])

    f
end
latency_plot(; probs=0.0001:0.00001:0.1)