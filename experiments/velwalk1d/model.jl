using Gen, Distributions
# Include the library exposing `Cat` and `LCat`
using ProbEstimates
ProbEstimates.use_perfect_weights!()

# Include some utilities for defining discrete probability distributions
include("../utils/modeling_utils.jl")
Positions() = 1:20
Vels() = -3:3
Bools() = [true, false]
VelStepStd() = 0.5
ObsStd() = 1.0
SwitchProb() = 0.0

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    vₜ ~ LCat(Vels())(unif(Vels()))
    return (xₜ, vₜ)
end
@gen (static) function step_latent_model(xₜ₋₁, vₜ₋₁)
    vₜ ~ LCat(Vels())(
        (1 - SwitchProb()) * discretized_gaussian(vₜ₋₁, VelStepStd(), Vels()) +
             SwitchProb()  * unif(Vels())
    )
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
    return (xₜ, vₜ)
end
@gen (static) function obs_model(xₜ, vₜ)
    obs ~ Cat(discretized_gaussian(xₜ, ObsStd(), Positions()))
    return (obs,)
end

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 2)
@load_generated_functions()