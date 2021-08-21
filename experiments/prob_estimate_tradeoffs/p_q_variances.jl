using Distributions
using CairoMakie
CairoMakie.activate!()

### Log(Fractional Variance) vs Latency
LatencyVals = LinRange(1.0, 500.0, 500)
Latency_Text = "Latency (ms)"
Latencies = (LatencyVals, Latency_Text)

AssemblySizes = [1, 10, 100, 1_000, 10_000]

function make_fracvar_plot(get_frac_var, p, title, (Xs, XText))
    f = Figure()
    ax = Axis(f[1, 1], xlabel=XText, ylabel="Log(Fractional Variance)"; title)
    plts = []
    for assemblysize in AssemblySizes
        push!(plts, lines!(ax, Xs, latency -> log(get_frac_var(latency, assemblysize, p))))
    end
    Legend(f[2, 1], plts, ["Assembly Size = $assemblysize neurons" for assemblysize in AssemblySizes])
    colsize!
    return f
end

direct_frac_var(size, latency, p) = direct_frac_var(size * latency, p)
recip_frac_var(size, latency, q) = recip_frac_var(size * latency * q, q)

direct_frac_var(K, p) = (1 - p)/(p * K) # 1/(K^2 * p^2) * var(Binomial(round(K), p))
recip_frac_var(K, q) = (1 - q) / K

is_weight_frac_var(size, latency, (p, q)) = indep_product_frac_var(
    direct_frac_var(size, latency, p),
    recip_frac_var(size, latency, q)
)
indep_product_frac_var(fracvars...) = prod(1 + fracvar for fracvar in fracvars) - 1

# make_plot(direct_frac_var, 0.1, "Fractional Variance of Direct Probability Estimates of P=0.1")
# make_plot(direct_frac_var, 0.01, "Fractional Variance of Direct Probability Estimates of P=0.01")
# make_plot(recip_frac_var, 0.1, "Fractional Variance of Reciprocal Probability Estimates for P=0.1")

# is_plt(p, q) = make_plot(is_weight_frac_var, (p, q), "Fractional Variance of P/Q Estimate for P=$p, Q=$q (P/Q = $(p/q))")

### Latency vs P -- for Log(Fractional Variance) = 0.
function direct_K_for_fracvar(p, fracvar)
    K = (1 - p)/(fracvar * p)
    @assert isapprox(fracvar, direct_frac_var(K, p)) "$fracvar | $(direct_frac_var(K, p))"
    return K
end
latency_for_direct_fracvar(p, fracvar, assemblysize) = direct_K_for_fracvar(p, fracvar) / assemblysize
function recip_K_for_fracvar(q, fracvar)
    K = (1 - q) / fracvar
    @assert isapprox(fracvar, recip_frac_var(K, q))
    return K
end
latency_for_recip_fracvar(p, fracvar, assemblysize) = recip_K_for_fracvar(p, fracvar) / (assemblysize * p)
function latency_for_isweight_fracvar(p, q, fracvar, assemblysize)
    a = fracvar; b = -((1-p)/p + (1-q)/q); c = -(1-p)/p * (1-q)/q
    LA = (-b + sqrt(b^2 - 4*a*c))/(2*a)
    latency = LA/assemblysize
    @assert isapprox(fracvar, is_weight_frac_var(assemblysize, latency, (p, q))) "$fracvar | $(is_weight_frac_var(assemblysize, latency, (p, q)))"
    return latency
end

function make_latency_p_plot(get_K, fracvar, est_string; xlabel="P")
    f = Figure()
    ax = Axis(f[1, 1]; xlabel, ylabel="Expected Latency (ms)", yscale=log10, xscale=log10, title="Expected Latency to achieve Fractional Variance = $fracvar for $est_string")
    ps = LinRange(0.001, 0.999, 200)
    plts = []
    for assemblysize in AssemblySizes
        push!(plts, lines!(ax, ps, p -> get_K(p, fracvar, assemblysize)))
    end
    Legend(f[2, 1], plts, ["Assembly Size = $assemblysize neurons" for assemblysize in AssemblySizes])
    colsize!
    return f
end
# make_latency_p_plot(latency_for_direct_fracvar, 1., "direct probability estimate")
# make_latency_p_plot(latency_for_recip_fracvar, 1., "reciprocal probability estimate", xlabel="Q")
# make_latency_p_plot((p, F, a) -> latency_for_isweight_fracvar(p, 0.1, F, a), 1, "IS weight w/ Q=0.1")
let Q = 0.1; make_latency_p_plot((p, F, a) -> latency_for_isweight_fracvar(p, Q, F, a), 0.1, "IS weight w/ Q=$Q"); end;
# make_latency_p_plot((p, F, a) -> latency_for_isweight_fracvar(p, 0.1, F, a), 10., "IS weight w/ Q=0.1")
# make_latency_p_plot((p, F, a) -> latency_for_isweight_fracvar(p, 0.1, F, a), 0.01, "IS weight w/ Q=0.1")
#=
TODOs:
- Plot with fixed latency + assembly size, but probability changing on the axis
- Plot IS weight fractional variance
=#