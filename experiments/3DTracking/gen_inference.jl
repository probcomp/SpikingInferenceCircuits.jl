using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
using ProbEstimates

include("model.jl")
ProbEstimates.use_perfect_weights!()
#ProbEstimates.use_noisy_weights!()

model = @DynamicModel(initial_model, step_model, obs_model, 10)
initial_proposal_compiled = @compile_initial_proposal(initial_proposal, 2)
step_proposal_compiled = @compile_step_proposal(step_proposal, 10, 2)
#step_proposal_compiled = @compile_step_proposal(step_model, 9, 2)
#initial_proposal_compiled = @compile_initial_proposal(initial_model, 2)

@load_generated_functions()

NSTEPS = 15
NPARTICLES = 100

tr = simulate(model, (NSTEPS,))

x_traj = [(:steps => i => :latents => :xₜ => :val, X_init + i) for i in 1:NSTEPS]
y_traj = [(:steps => i => :latents => :yₜ => :val, Y_init + i) for i in 1:NSTEPS]
z_traj = [(:steps => i => :latents => :zₜ => :val, Z_init) for i in 1:NSTEPS]
vx_traj = [(:steps => i => :latents => :vxₜ => :val, 1) for i in 1:NSTEPS]
vy_traj = [(:steps => i => :latents => :vyₜ => :val, 1) for i in 1:NSTEPS]
vz_traj = [(:steps => i => :latents => :vzₜ => :val, 0) for i in 1:NSTEPS]

# tr, w = generate(model, (NSTEPS,), choicemap(
#     (:init => :latents => :xₜ => :val, X_init),
#     (:init => :latents => :yₜ => :val, Y_init),
#     (:init => :latents => :zₜ => :val, Z_init),
#     (:init => :latents => :vxₜ => :val, 1),
#     (:init => :latents => :vyₜ => :val, 1), 
#     (:init => :latents => :vzₜ => :val, 0),
#     x_traj...,
#     y_traj...,
#     z_traj...,
#     vz_traj...,
#     vx_traj...,
#     vy_traj...
# ))


  # to constrain a step:
#  (:steps => t => :latents => :x, val)
#)

observations = get_dynamic_model_obs(tr)

(unweighted_traces_at_each_step, _) = dynamic_model_smc(
    model, observations,
    ch -> (ch[:obs_θ => :val], ch[:obs_ϕ => :val]),
    initial_proposal_compiled, step_proposal_compiled,
    NPARTICLES, # n particles
    ess_threshold=NPARTICLES)

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

