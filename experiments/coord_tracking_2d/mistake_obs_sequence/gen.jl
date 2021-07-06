using Gen, Distributions
includet("../../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

includet("../model_proposal.jl")

# Convert initial, step, and obs model into a generative function
# which simulates the resulting dynamic probabilistic program for T steps.
model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 4) # 4 = num latent variables in model
# Convert the proposals into proposals for the overall dynamic model:
proposal_init = @compile_initial_proposal(initial_proposal, 2) # 2 = num observed variables
proposal_step = @compile_step_proposal(step_proposal, 4, 2) # 4 = num latent variables; 2 = num observed variables
@load_generated_functions()

observations = [(1, 1), (2, 2), (3, 3), (4, 4), (3, 3), (6, 6)]
(x1, y1), rest = Iterators.peel(observations)
tr, _ = generate(model, (length(observations) - 1,), choicemap(
    (:init => :obs => :obsx => :val, x1), (:init => :obs => :obsy => :val, y1),
    Iterators.flatten(
        ((:steps => t => :obs => :obsx => :val, x), (:steps => t => :obs => :obsy => :val, y))
        for (t, (x, y)) in enumerate(rest)
    )...
))
obss = get_dynamic_model_obs(tr)

ProbEstimates.set_latency!(20)
ProbEstimates.set_assembly_size!(10)
ProbEstimates.use_noisy_weights!()
@assert ProbEstimates.K_fwd() == 40 && ProbEstimates.K_recip() == 4

tr_to_latents(tr, t::Int) =
    t > 0 ? tr_to_latents(tr, :steps => t => :latents) : tr_to_latents(tr, :init => :latents)

tr_to_latents(tr, addr) =
    let ch = get_submap(get_choices(tr), addr)
        (pos=(x=ch[:xₜ => :val], y=ch[:yₜ => :val]), vel=(x=ch[:vxₜ => :val], y=ch[:vyₜ => :val]))
    end

function get_inferences()
    (unweighted_inferences, weighted_inferences) = dynamic_model_smc(
        model, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
        proposal_init, proposal_step, 6
    )
    inferences = [
        [tr_to_latents(tr, t - 1) for tr in trs]
        for (t, trs) in enumerate(unweighted_inferences)
    ]
    return (inferences, weighted_inferences)
end

inferences, _ = get_inferences()
nothing

# From here I manually examined `inferences` to check that it looks like
# the inferences looked as I hoped