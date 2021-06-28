using Base: Int64
include("../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels: @DynamicModel, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc

include("model.jl")

model = @DynamicModel(initial_model, step_model, obs_model, 9)
step_proposal = @compile_step_proposal(step_proposal, 9, 2)
@load_generated_functions()

NSTEPS = 5
NPARTICLES = 20
tr = simulate(model, (NSTEPS,))
observations = get_dynamic_model_obs(tr)

(unweighted_traces_at_each_step, _) = dynamic_model_smc(
    model, observations,
    ch -> (ch[:obs_θ => :val], ch[:obs_ϕ => :val]),
    initial_proposal, step_proposal,
    NPARTICLES # n particles
)


heatmap_pf_results(unweighted_traces_at_each_step, tr, NSTEPS)

# unweighted_traces_at_each_step looks like
# [
    # [particle1trace, particle2trace, ...] # for timestep 1
    # [particle1trace, particle2trace, ...] # for timestep 2
    # [particle1trace, particle2trace, ...] # for timestep 3
# ]
# where the traces are the traces we have after resampling

# by a "trace for timestep T", I mean a trace which has choices
# for every timestep up to and including T


