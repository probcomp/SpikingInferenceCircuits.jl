module ProbEstimates
using Gen
using Distributions
using DiscreteIRTransforms

include("categorical.jl")

include("prob_est_hyperparameters.jl")

K_fwd() = Latency() * AssemblySize() * MaxRate() |> Int ∘ round
K_recip() = Latency() * (MaxRate() * MinProb()) * AssemblySize() |> Int ∘ round

acceptable_p_error(n_vars_in_model, err_constant=10) = K_fwd() > log(err_constant * n_vars_in_model) / MinProb()

# defines `fwd_prob_estimate` and `recip_prob_estimate`
# and utils for changing how they work to use perfect vs noisy weights.
include("weight_mode_switching.jl")

include("pseudomarginal_dist.jl")

export LCat, Cat, PseudoMarginalizedDist

include("compilation_compatibility.jl")

include("to_single_line.jl")

end # module