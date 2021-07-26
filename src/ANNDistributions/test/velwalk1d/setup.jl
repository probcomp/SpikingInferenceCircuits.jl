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