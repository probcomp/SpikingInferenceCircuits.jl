using DynamicModels
include("model.jl")
include("inference.jl")
include("visualize.jl")

ProbEstimates.DoRecipPECheck() = false

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 1)
@load_generated_functions()

# tr, _ = generate(model, (10,))

function obs_pos_enumerated_figure(tr)
    enumerated_weights = enumeration_bayes_filter_from_groundtruth(
        tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),)
    ) |> nest_all_addrs_at_val |> collect

    (fig, t) = obs_pos_inferences_figure(tr, [(
        "Exact posterior probabilities",
        t -> exp.(enumerated_weights[t + 1]) |> normalize
    )])
    
    return (fig, t)
end

function x_counts(trs, t)
    counts = Int[0 for _ in Positions()]
    for tr in trs
        counts[latents_choicemap(tr, t)[:xₜ => :val]] += 1
    end
    return counts
end
function obs_pos_priorsmc_figure(tr; n_particles=1_000)
    (unweighted_trs, _) = smc_from_prior(tr, n_particles)
    
    (fig, t) = obs_pos_inferences_figure(tr, [(
        "Inferred Probabilities via SMC from Prior with $n_particles Particles & full resampling",
        t -> x_counts(unweighted_trs[t + 1], t) |> normalize
    )])
    
    return (fig, t)
end

make_true_2d_posterior_figure(tr) = make_2d_posterior_figure(tr,
    enumeration_bayes_filter_from_groundtruth(
            tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),)
        ) |> DynamicModels.nest_all_addrs_at_val |> collect |> x->map(x->normalize(exp.(x)), x);
        inference_method_str="Posterior from exact Bayes filter."
)
function make_smc_figure(smcfn, tr; n_particles, proposalstr)
    (unweighted_trs, _) = smcfn(tr, n_particles)
    probs = [x_counts(unweighted_trs[t + 1], t) |> normalize for t=0:(get_args(tr)[1])]
    make_2d_posterior_figure(tr, probs; inference_method_str="Approximate posterior from $n_particles particle SMC $proposalstr")
end
make_smcprior_2d_posterior_figure(tr; n_particles=1_000) = make_smc_figure(smc_from_prior, tr; n_particles, proposalstr="proposing from prior")
make_smcexact_2d_posterior_figure(tr; n_particles=10)    = make_smc_figure(smc_exact_proposal, tr; n_particles, proposalstr="proposing from exact posterior")

make_smc_prior_exactrejuv_2d_posterior_figure(tr; n_particles=10) =
    make_smc_figure(prior_smc_exact_rejuv, tr; n_particles, proposalstr="\nproposing from prior + using gibbs rejuvenation")

# make_smc_prior_exactrejuv_2d_posterior_figure(tr)

tr, _ = generate(model, (10,))
fig = make_smcexact_2d_posterior_figure(tr; n_particles=2)