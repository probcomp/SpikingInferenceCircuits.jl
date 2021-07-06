using CairoMakie
includet("model_proposal.jl")
includet("pm_obs_models.jl")

## A setting is a pair `(noisemode, latency_in_ms)`
# e.g. (:perfect, 50)  or  (:noisy, 50)
# [if :perfect, the latency value won't effect outcomes]
function apply_weight_setting!((noisemode, latency))
    current_setting = (ProbEstimates.weight_type(), ProbEstimates.Latency())
    ProbEstimates.set_weighttype_to!(noisemode)
    ProbEstimates.set_latency!(latency)
    return current_setting
end

prob_estimates(model, x, obs, n_samples) = [
    exp(assess(model, (x, 1), choicemap((:obsx => :val, obs)))[1])
    for _=1:n_samples
]

function prob_estimates_for_settings(x, obs, models, settings; n_samples)
    estimates = Dict()
    for setting in settings
        rst = apply_weight_setting!(setting)
        for (modelname, model) in models
            estimates[(modelname, setting)] = prob_estimates(model, x, obs, n_samples)
        end
        apply_weight_setting!(rst) # reset to previous setting
    end
    return estimates
end

function plot_variances(x, obs, latencies, models; n_samples)
    ests = prob_estimates_for_settings(x, obs, models,
        [(:noisy, lat) for lat in latencies]; n_samples
    )
    
    fig = Figure()
    ax = Axis(fig[1, 1],
        title="Obs Likelihood Estimates for x=$x, obs=$obs",
        ylabel="Std.Dev. of probability estimate",
        xlabel="Scoring latency (ms)"
    )
    
    for (modelname, _) in models
        scatter!(ax,
            latencies,
            [stddev(ests[(modelname, (:noisy, latency))]) for latency in latencies],
            label="$modelname"
        )
    end
    axislegend()
    return fig
end
stddev(args...; kwargs...) = sqrt(var(args...; kwargs...))
var(samples; expectation=sum(samples)/length(samples)) =
    sum((s - expectation)^2 for s in samples)/length(samples)

@load_generated_functions()

plot_variances(10, 13, 10:10:50, [
    ("Direct Distribution Encoding",   obs_model_direct),
    ("PM Proposing from Prior", obs_model_naive_pseudomarginalization),
    ("PM w/ Exact Proposal", obs_model_exact_pseudomarginalization),
    ("PM w/ Heuristic Proposal", obs_model_intermediate_pseudomarginalization)
]; n_samples=2000)