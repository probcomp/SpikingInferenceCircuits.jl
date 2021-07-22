using DynamicModels
include("model.jl")
include("pm_model.jl")
include("inference.jl")
include("visualize.jl")
ProbEstimates.DoRecipPECheck() = false

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 2)
@load_generated_functions()

include("run_utils.jl")

tr, _ = generate(model, (10,));
make_smc_approx_mhrejuv_2d_posterior_figure(tr)