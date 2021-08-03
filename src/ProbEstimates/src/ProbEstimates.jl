module ProbEstimates
using Gen
using Distributions

include("hyperparameters.jl")

recip_truncate(probs) = TruncateRecipDists() ? truncate(probs) : probs
fwd_truncate(probs)   = TruncateFwdDists()   ? truncate(probs) : probs
function truncate(pvec)
    if !isprobvec(pvec)
        error("pvec = $pvec is not a probability vector")
    end
    mininvec = minimum(p for p in pvec if p != 0)
    if mininvec ≥ MinProb()
        return pvec
    else
        first_to_truncate = findfirst(pvec .== mininvec)
        return truncate(
            normalize([i == first_to_truncate ? 0. : p for (i, p) in enumerate(pvec)])
        )
    end
end
normalize(vec) = vec/sum(vec)

include("categorical.jl")

K_fwd() = Latency() * AssemblySize() * MaxRate() |> Int ∘ round
K_recip() = Latency() * (MaxRate() * MinProb()) * AssemblySize() |> Int ∘ round

acceptable_p_error(n_vars_in_model, err_constant=10) = K_fwd() > log(err_constant * n_vars_in_model) / MinProb()

# defines `fwd_prob_estimate` and `recip_prob_estimate`
# and utils for changing how they work to use perfect vs noisy weights.
include("weight_mode_switching.jl")

include("pseudomarginal_dist.jl")

export LCat, Cat, PseudoMarginalizedDist

include("compilation_compatibility.jl")

include("normalize_weights.jl")

# overwrite methods in DynamicModels so that inference uses the noise model
include("dynamic_models_overwrites.jl")

end # module