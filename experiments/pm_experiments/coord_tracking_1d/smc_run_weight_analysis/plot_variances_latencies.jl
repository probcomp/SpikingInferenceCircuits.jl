using Gen, Distributions
includet("../../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

includet("model_proposal.jl")
includet("naive_inference.jl")

NSTEPS = 10
model_direct   = @DynamicModel(initial_latent_model, step_latent_model, direct_obs_model, 2)
model_pm_prior = @DynamicModel(initial_latent_model, step_latent_model, obs_model_naive_pseudomarginalization, 2)
model_pm_exact = @DynamicModel(initial_latent_model, step_latent_model, obs_model_exact_pseudomarginalization, 2)
model_pm_inter = @DynamicModel(initial_latent_model, step_latent_model, obs_model_intermediate_pseudomarginalization, 2)

proposal_init = @compile_initial_proposal(initial_proposal, 1)
proposal_step = @compile_step_proposal(step_proposal, 2, 1)
@load_generated_functions()

includet("plotting_and_analysis.jl")

statistics = get_noisy_weight_statistics(
    model_direct,
    [model_pm_prior, model_pm_exact, model_pm_inter],
    latencies, # TODO
    compute_std_over_weight
)