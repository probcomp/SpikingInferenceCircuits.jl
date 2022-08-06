### Single-estimate fractional variance ###
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

### Distributed probability estimate fractional variance ###
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

### Single-pixel latency at fixed fractional variance ###
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

### Default hyperparameters ###
default_hyperparams() = (
    latency        = 5,   # ms
    frequency      = 0.2, # spikes per ms
    neuron_budget = 200
)
get_k(hyperparams) = hyperparams.latency * hyperparams.neuron_budget * hyperparams.frequency