using DynamicModels, ANNDistributions
import BSON
using ANNDistributions.Flux
include("../../../../experiments/velwalk1d/model.jl")
include("../../../../experiments/velwalk1d/inference.jl")
include("../../../../experiments/velwalk1d/visualize.jl")

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 2)
@load_generated_functions()

# Step proposal using ANN
ann = BSON.load("model-checkpoint.bson")[:model]
VelStepDist = ANN_LCPT((Positions(), Vels(), Positions()), Vels(), ann)
@gen (static) function _ann_step_proposal(xₜ₋₁, vₜ₋₁, obs)
    vₜ ~ VelStepDist(xₜ₋₁, vₜ₋₁, obs)
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
end

exact_init_proposal = @compile_initial_proposal(_exact_init_proposal, 1)
ann_step_proposal = @compile_step_proposal(_ann_step_proposal, 2, 1)
@load_generated_functions()

smc_ann_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, ann_step_proposal)

include("../../../../experiments/velwalk1d/run_utils.jl")

make_smcann_2d_posterior_figure(tr; n_particles=10) =
    make_smc_figure(smc_ann_proposal, tr; n_particles, proposalstr="proposing using ANN")

tr, _ = generate(model, (10,));
make_smcann_2d_posterior_figure(tr)