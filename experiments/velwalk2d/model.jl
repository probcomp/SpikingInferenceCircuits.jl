using Gen, Distributions
# Include the library exposing `Cat` and `LCat`
using ProbEstimates
ProbEstimates.use_perfect_weights!()

# Include some utilities for defining discrete probability distributions
includet("../utils/modeling_utils.jl")
Positions() = 1:10
Vels() = -2:2
VelStepStd() = 0.5
ObsStd() = 0.5
SwitchProb() = 0.

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    yₜ ~ Cat(unif(Positions()))
    vxₜ ~ LCat(Vels())(unif(Vels()))
    vyₜ ~ LCat(Vels())(unif(Vels()))
    return (xₜ, yₜ, vxₜ, vyₜ)
end

v_step_dist(vₜ₋₁) = (
    (1 - SwitchProb()) * discretized_gaussian(vₜ₋₁, VelStepStd(), Vels()) +
         SwitchProb()  * unif(Vels())
)
@gen (static) function step_latent_model(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁)
    vxₜ ~ LCat(Vels())(v_step_dist(vxₜ₋₁))
    vyₜ ~ LCat(Vels())(v_step_dist(vyₜ₋₁))

    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, Positions()))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, Positions()))

    return (xₜ, yₜ, vxₜ, vyₜ)
end
@gen (static) function obs_model(xₜ, yₜ, vxₜ, vyₜ)
    obsx ~ Cat(discretized_gaussian(xₜ, ObsStd(), Positions()))
    obsy ~ Cat(discretized_gaussian(yₜ, ObsStd(), Positions()))
    return (obsx, obsy)
end