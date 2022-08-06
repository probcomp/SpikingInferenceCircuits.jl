using Gen, Distributions
# Include the library exposing `Cat` and `LCat`
using ProbEstimates
ProbEstimates.use_perfect_weights!()
ProbEstimates.DoRecipPECheck() = false

# Include some utilities for defining discrete probability distributions
include("../utils/modeling_utils.jl")
Positions() = 1:10
Vels() = -2:2
Vels2D() = [(x, y) for x in Vels() for y in Vels()]
VelStepStd() = 0.5
ObsStd() = 0.5
SwitchProb() = 0.1

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    yₜ ~ Cat(unif(Positions()))
    vₜ ~ LCat(Vels2D())(unif(Vels2D()))
    (vx, vy) = vₜ
    vxₜ ~ LCat(Vels())(onehot(vx, Vels()))
    vyₜ ~ LCat(Vels())(onehot(vy, Vels()))

    return (xₜ, yₜ, vxₜ, vyₜ)
end

function vel_step_dist(vₜ₋₁)
    (vxₜ₋₁, vyₜ₋₁) = vₜ₋₁
    vxₜ_dist = discretized_gaussian(vxₜ₋₁, VelStepStd(), Vels())
    vyₜ_dist = discretized_gaussian(vyₜ₋₁, VelStepStd(), Vels())
    walk_dist = [px * py for px in vxₜ_dist for py in vyₜ_dist]
    return (
             SwitchProb()  * unif(Vels2D()) +
        (1 - SwitchProb()) * walk_dist
    )
end
@gen (static) function step_latent_model(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁)
    vₜ ~ LCat(Vels2D())(vel_step_dist((vxₜ₋₁, vyₜ₋₁)))
    (vx, vy) = vₜ
    vxₜ ~ LCat(Vels())(onehot(vx, Vels()))
    vyₜ ~ LCat(Vels())(onehot(vy, Vels()))

    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, Positions()))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, Positions()))

    return (xₜ, yₜ, vxₜ, vyₜ)
end
@gen (static) function obs_model(xₜ, yₜ, vxₜ, vyₜ)
    obsx ~ Cat(discretized_gaussian(xₜ, ObsStd(), Positions()))
    obsy ~ Cat(discretized_gaussian(yₜ, ObsStd(), Positions()))
    return (obsx, obsy)
end