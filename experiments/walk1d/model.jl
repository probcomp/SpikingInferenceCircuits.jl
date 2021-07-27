using Gen, Distributions
# Include the library exposing `Cat` and `LCat`
using ProbEstimates
ProbEstimates.use_perfect_weights!()

# Include some utilities for defining discrete probability distributions
include("../utils/modeling_utils.jl")
Positions() = 1:20
Bools() = [true, false]
StepStd() = 3.
ObsStd() = 2.0

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    return (xₜ,)
end
@gen (static) function step_latent_model(xₜ₋₁)
    xₜ ~ Cat(discretized_gaussian(xₜ₋₁, StepStd(), Positions()))
    return (xₜ,)
end
@gen (static) function obs_model(xₜ)
    obs ~ Cat(discretized_gaussian(xₜ, ObsStd(), Positions()))
    return (obs,)
end