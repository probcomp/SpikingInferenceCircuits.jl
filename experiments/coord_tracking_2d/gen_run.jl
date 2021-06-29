using Gen, Distributions
includet("../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

includet("model_proposal.jl")

NSTEPS = 10
# Convert initial, step, and obs model into a generative function
# which simulates the resulting dynamic probabilistic program for T steps.
model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 4) # 4 = num latent variables in model
# Convert the proposals into proposals for the overall dynamic model:
proposal_init = @compile_initial_proposal(initial_proposal, 2) # 2 = num observed variables
proposal_step = @compile_step_proposal(step_proposal, 4, 2) # 4 = num latent variables; 2 = num observed variables
@load_generated_functions()

# Get a trace to run inference on, and extract the observations from that trace
tr = simulate(model, (NSTEPS,))
obss = get_dynamic_model_obs(tr)

# Run inference using Gen floating-point score calculations
ProbEstimates.use_perfect_weights!()
(unweighted_inferences, weighted_inferences) = dynamic_model_smc(
    model, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
    proposal_init, proposal_step, 20
)
# unweighted_inferences is a vector of vectors [[trace1_time1, ... traceN_time1], [trace1_time2, .., traceN_time2], ...,]
# giving the trace for each particle at the given timestep.  (The trace will be a full trace containing all decisions
# up to that timestep.)  These are the traces _after_ the resampling step.  (By default all particles are resampled at each
# step, and so these are unweighted particle clouds).
# `weighted_inferences` is a vector of vectors of pairs (trace, logweight) giving the trace and its log weight
# before the resampling step.

# Run in Gen with noisy weight estimates from the same distribution
# as the weight estimates in the spiking neural network:
ProbEstimates.use_noisy_weights!()
(noisy_inference_unweighted_traces, noisy_inference_weighted_traces) = dynamic_model_smc(
    model, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
    proposal_init, proposal_step, 20
)