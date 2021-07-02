using Base: Int64
include("../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc

include("model.jl")

model = @DynamicModel(initial_model, step_model, obs_model, 8)
step_proposal = @compile_step_proposal(step_proposal, 8, 2)
initial_proposal = @compile_initial_proposal(initial_proposal, 2)
@load_generated_functions()

NSTEPS = 10
NPARTICLES = 20
#tr = simulate(model, (NSTEPS,))

tr, w = generate(model, (NSTEPS,), choicemap(
    (:init => :latents => :moving_in_depthₜ, true),
    (:init => :latents => :xₜ => :val, 5),
    (:init => :latents => :yₜ => :val, -5),
    (:init => :latents => :heightₜ => :val, 10),
    (:init => :latents => :vₜ => :val, 1)))

  # to constrain a step:
#  (:steps => t => :latents => :x, val)
#)

observations = get_dynamic_model_obs(tr)

(unweighted_traces_at_each_step, _) = dynamic_model_smc(
    model, observations,
    ch -> (ch[:obs_θ => :val], ch[:obs_ϕ => :val]),
    initial_proposal, step_proposal,
    NPARTICLES, # n particles
    ess_threshold=NPARTICLES
)

#OK next step is figuring out which particles are moving in depth vs not

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


#tr_init = simulate(model, (0,))
#proposed_choices, _ = propose(step_proposal, (tr_init, 0.0, 0.0))
#[propose(step_proposal, (tr_init, 0.0, 0.0))[1][:steps => 1 => :latents => :moving_in_depthₜ] for i in 1:200]

