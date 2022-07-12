using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
using ProbEstimates


include("model.jl")
include("ab_viz.jl")
ProbEstimates.use_perfect_weights!()
#ProbEstimates.use_noisy_weights!()

model = @DynamicModel(initial_model, step_model, obs_model, 9)
initial_proposal_compiled = @compile_initial_proposal(initial_proposal, 2)
step_proposal_compiled = @compile_step_proposal(step_proposal, 9, 2)

@load_generated_functions()

NSTEPS = 14
NPARTICLES = 20
cmap = make_deterministic_trace()
tr, w = generate(model, (NSTEPS,), cmap)
observations = get_dynamic_model_obs(tr)

(unweighted_traces_at_each_step, _) = dynamic_model_smc(
    model, observations,
    ch -> (ch[:obs_ϕ => :val], ch[:obs_θ => :val]),
    initial_proposal_compiled, step_proposal_compiled,
    NPARTICLES, # n particles
    ess_threshold=NPARTICLES)

#heatmap_pf_results(unweighted_traces_at_each_step, tr, NSTEPS)
animate_pf_results(unweighted_traces_at_each_step, tr)
render_static_trajectories(unweighted_traces_at_each_step, tr)
println([get_score(t) for t in unweighted_traces_at_each_step[1]])
render_obs_from_particles(unweighted_traces_at_each_step, 10);


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

