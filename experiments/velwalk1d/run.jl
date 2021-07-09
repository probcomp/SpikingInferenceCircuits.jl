includet("../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

includet("model.jl")
includet("inference.jl")
includet("visualize.jl")

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 2)
@load_generated_functions()

function get_x_probs(tr)
    logweight_grids = enumeration_bayes_filter_from_groundtruth(
            tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),vₜ=Vels())
        ) |> nest_all_addrs_at_val
    weight_grids = [exp.(logweight_grid) for logweight_grid in logweight_grids]
    weight_vecs = [sum(grid, dims=2) for grid in weight_grids]
    return [normalize(weights) for weights in weight_vecs]
end

make_smcprior_2d_posterior_figure(tr; n_particles=1_000) = make_smc_figure(smc_from_prior, tr; n_particles, proposalstr="proposing from prior")

make_true_2d_posterior_figure(tr) = make_2d_posterior_figure(tr, get_x_probs(tr);
    inference_method_str="Posterior from exact Bayes filter."
)
make_smcexact_2d_posterior_figure(tr; n_particles=10)    = make_smc_figure(smc_exact_proposal, tr; n_particles, proposalstr="proposing from exact posterior")
make_smc_prior_exactrejuv_2d_posterior_figure(tr; n_particles=10) =
    make_smc_figure(prior_smc_exact_rejuv, tr; n_particles, proposalstr="\nproposing from prior + using gibbs rejuvenation")

tr, _ = generate(model, (10,));
make_smcexact_2d_posterior_figure(tr)