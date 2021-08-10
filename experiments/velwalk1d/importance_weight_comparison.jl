using DynamicModels
include("model.jl")
# include("pm_model.jl")
include("inference.jl")
include("visualize.jl")
ProbEstimates.DoRecipPECheck() = false

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 2)
@load_generated_functions()

true_step_importance_weight(pre_update_tr, post_update_tr, proposal) =
    ProbEstimates.with_weight_type(:perfect) do 
        T = get_args(post_update_tr)[1]
        @assert get_args(pre_update_tr)[1] == T - 1
        proposed_choices = get_selected(get_choices(post_update_tr), select(:steps => T => :latents))
        obs = post_update_tr[:steps => T => :obs]
        propose_score, _ = assess(proposal, (pre_update_tr, obs...), proposed_choices)
        newtr, assess_score, _, _  = update(pre_update_tr, (T,), (UnknownChange(),), merge(
            proposed_choices, get_selected(get_choices(post_update_tr), select(:steps => T => :obs))
        ))
        if !ProbEstimates.choicemaps_have_equal_values(get_choices(newtr), get_choices(post_update_tr))
            println("proposed choices:")
            display(proposed_choices)
            println()
            println("new tr:")
            display(get_choices(newtr))
            println()
            println("post update tr:")
            display(get_choices(post_update_tr))
            error()
        end

        assess_score - propose_score
    end

function true_log_stepweights_to_noisy(weighted_trs, proposal)
    true_logweight_to_logweight = Dict{Float64, Vector{Float64}}()
    for weighted_trs_at_timestep in Iterators.drop(weighted_trs, 1)
        for (weighted_new_tr, weight) in weighted_trs_at_timestep
            T = get_args(weighted_new_tr)[1]
            pre_update_tr, _, _, _ = update(weighted_new_tr, (T-1,), (UnknownChange(),), EmptyChoiceMap())
            true_weight = true_step_importance_weight(pre_update_tr, weighted_new_tr, proposal)
            push!(get!(true_logweight_to_logweight, true_weight, []), weight)
        end
    end
    return true_logweight_to_logweight
end

logavg(vec) = logsumexp(vec) - log(length(vec))
function min_max_fraction(v, frac)
    srt = sort(v)
    l = length(v)
    idxs = [
        max(1, Int(floor(l * frac))),
        min(l, Int(ceil(l * frac)))
    ]
    [srt[i] for i in idxs]
end

function weight_comparison_figure(logweight_to_noisy_logweights, additional_str="", legend_pos=:rb)
	log_true_weights = collect(keys(logweight_to_noisy_logweights))
	log_avg_weights = [logavg(logweight_to_noisy_logweights[w]) for w in log_true_weights]
	# upper_errors = [
	# 	min_max_fraction(logweight_to_noisy_logweights[log_true], 0.95)[2] - log_true
    #     for log_true in log_true_weights
	# ]
    # lower_errors = [
	# 	log_true - min_max_fraction(logweight_to_noisy_logweights[log_true], 0.05)[1]
    #     for log_true in log_true_weights
	# ]
    
	fig = Figure(resolution=(800, 450))
	ax = fig[1,1] = Axis(fig,
		title="Accuracy of Approximate Importance Weights from Dynamically Weighted Spike Code",
			xlabel="Log importance weight for SMC transition",
			ylabel="Sampled DWSC importance weights for transition"
	)
	
	# scat = scatter!(
	# 	log_true_weights, log_avg_weights, color=:black
	# )
    xs = reduce(vcat, [k for _ in v] for (k, v) in logweight_to_noisy_logweights)
    ys = reduce(vcat, v for (_, v) in logweight_to_noisy_logweights)
    scat = scatter!(
        ax, xs, ys, color=:black
    )
	# errbars = errorbars!(
	# 	log_true_weights,
	# 	log_avg_weights,
	# 	lower_errors,
	# 	upper_errors,
	# 	color=:red
	# )
	
	minval = minimum(log_true_weights)
	maxval = maximum(log_true_weights)
	
	line = lines!([minval, maxval], [minval, maxval])
	
	fig
end

gt_tr, _ = generate(model, (10,))

ProbEstimates.use_noisy_weights!()

n_particles() = 10
(unweighted_trs, weighted_trs) = smc_approx_proposal(gt_tr, n_particles());

dict = true_log_stepweights_to_noisy(weighted_trs, approx_step_proposal)

f = weight_comparison_figure(dict, "", :tr)