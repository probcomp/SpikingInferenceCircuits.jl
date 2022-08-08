using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
import DynamicModels
using ProbEstimates

includet("../model.jl")
includet("../ab_viz.jl")
include("deferred_inference.jl")

println("MinProb() = $(MinProb())")
# ProbEstimates.use_perfect_weights!()
ProbEstimates.use_noisy_weights!()
ProbEstimates.set_assembly_size!(30)
ProbEstimates.set_latency!(300)
ProbEstimates.UseLowPrecisionMultiply() = false
ProbEstimates.MultAssemblySize() = 200
ProbEstimates.MaxRate() = 100 # Hz

model = @DynamicModel(initial_model, step_model, obs_model, 9)
initial_proposal_compiled = @compile_initial_proposal(initial_proposal, 2)
step_proposal_compiled = @compile_step_proposal(step_proposal, 9, 2)
two_timestep_proposal_dumb = @compile_2timestep_proposal(initial_proposal, step_proposal, 9, 2)

@load_generated_functions()

NSTEPS = 3
NPARTICLES = 10
cmap = get_selected(make_deterministic_trace(), select(:init, :steps => 1, :steps => 2, :steps => 3))
tr, w = generate(model, (NSTEPS,), cmap)
observations = get_dynamic_model_obs(tr);

final_particle_set = []

for i in 1:100
    # try
        (unweighted_traces_at_each_step, weighted_traces) = deferred_dynamic_model_smc(
            model, (observations[1], observations[2][1:3]),
            ch -> (ch[:obs_ϕ => :val], ch[:obs_θ => :val]),
            two_timestep_proposal_dumb,
            # propose_first_two_timesteps_smart,
            step_proposal_compiled,
            NPARTICLES, # n particles
            ess_threshold=NPARTICLES
        );

        weights = map(x -> x[2], weighted_traces[end])
        particles = map(x -> x[1], weighted_traces[end])
        pvec = normalize(exp.(weights .- logsumexp(weights)))
        if !isprobvec(pvec)
            continue
        else
            sample = Gen.categorical(pvec)
            push!(final_particle_set, particles[sample])
        end

    # catch
    #     continue
    # end
end
length(final_particle_set)

# animate_pf_results(final_particle_set, tr, true)
# animate_pf_results(final_particle_set, tr, false)
# render_static_trajectories(final_particle_set, tr, true)
render_static_trajectories(final_particle_set, tr, false)
# final_scores = [get_score(t) for t in final_particle_set]
# final_probs = normalize(exp.(final_scores .- logsumexp(final_scores)))
render_obs_from_particles(final_particle_set, 100; do_obs=false);

# plot_full_choicemap(final_particle_set)


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
