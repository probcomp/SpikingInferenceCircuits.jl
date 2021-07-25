using DynamicModels
include("model.jl")
# include("pm_model.jl")
include("inference.jl")
include("visualize.jl")
ProbEstimates.DoRecipPECheck() = false

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 2)
@load_generated_functions()

function get_enumeration_grids(tr)
    logweight_grids = enumeration_bayes_filter_from_groundtruth(
            tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),vₜ=Vels())
        ) |> DynamicModels.nest_all_addrs_at_val
    weight_grids = [exp.(logweight_grid) for logweight_grid in logweight_grids]
    return weight_grids
end

function x_vel_counts(trs, t)
    counts = Int[0 for _ in Positions(), _ in Vels()]
    for tr in trs
        counts[
            latents_choicemap(tr, t)[:xₜ => :val],
            latents_choicemap(tr, t)[:vₜ => :val] - first(Vels()) + 1
        ] += 1
    end
    return counts
end

function make_smc_figure(smcfn, tr; n_particles, proposalstr)
    (unweighted_trs, _) = smcfn(tr, n_particles)
    probs = [x_vel_counts(unweighted_trs[t + 1], t) |> normalize for t=0:(get_args(tr)[1])]
    make_2d_posterior_figure(tr, probs; inference_method_str="Approximate posterior from $n_particles particle SMC $proposalstr")
end

make_smcprior_2d_posterior_figure(tr; n_particles=1_000) = make_smc_figure(smc_from_prior, tr; n_particles, proposalstr="proposing from prior")

make_true_2d_posterior_figure(tr) = make_2d_posterior_figure(tr, get_enumeration_grids(tr);
    inference_method_str="Posterior from exact Bayes filter."
)
make_smcexact_2d_posterior_figure(tr; n_particles=10) =
    make_smc_figure(smc_exact_proposal, tr; n_particles, proposalstr="proposing from exact posterior")
make_smcapprox_2d_posterior_figure(tr; n_particles=10) =
    make_smc_figure(smc_approx_proposal, tr; n_particles, proposalstr="\nproposing from efficiently-encoded approximate posterior")

make_smc_prior_exactrejuv_2d_posterior_figure(tr; n_particles=10) =
    make_smc_figure(prior_smc_exact_rejuv, tr; n_particles, proposalstr="\nproposing from prior + using gibbs rejuvenation")

# tr, _ = generate(model, (10,));
# make_smcexact_2d_posterior_figure(tr)