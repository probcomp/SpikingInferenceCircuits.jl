using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
using ProbEstimates


include("model.jl")
include("ab_viz.jl")

print(MinProb())
#ProbEstimates.use_perfect_weights!()
ProbEstimates.use_noisy_weights!()
ProbEstimates.set_assembly_size!(10)
ProbEstimates.set_latency!(300)
ProbEstimates.UseLowPrecisionMultiply() = false
ProbEstimates.MultAssemblySize() = 200
ProbEstimates.MaxRate() = 600 # Hz

model = @DynamicModel(initial_model, step_model, obs_model, 9)
initial_proposal_compiled = @compile_initial_proposal(initial_proposal, 2)
step_proposal_compiled = @compile_step_proposal(step_proposal, 9, 2)

@load_generated_functions()

NSTEPS = 10
NPARTICLES = 100
cmap = make_deterministic_trace()
tr, w = generate(model, (NSTEPS,), cmap)
observations = get_dynamic_model_obs(tr);

final_particle_set = []

for i in 1:100
    try
        (unweighted_traces_at_each_step, weighted_traces) = dynamic_model_smc(
            model, observations,
            ch -> (ch[:obs_ϕ => :val], ch[:obs_θ => :val]),
            initial_proposal_compiled, step_proposal_compiled,
            NPARTICLES, # n particles
            ess_threshold=NPARTICLES);
        scores = [get_score(t) for t in unweighted_traces_at_each_step[end]];
    
        if !any(map(isfinite, scores))
            continue
        end
        particle_sample = Gen.categorical(normalize(exp.(scores .- logsumexp(scores))))
        push!(final_particle_set, unweighted_traces_at_each_step[end][particle_sample])
    catch
        continue
    end
          
#    println(sum(normalize(exp.(scores .- logsumexp(scores)))))
  
end

# animate_pf_results(final_particle_set, tr, true)
# animate_pf_results(final_particle_set, tr, false)
render_static_trajectories(final_particle_set, tr, true)
render_static_trajectories(final_particle_set, tr, false)
final_scores = [get_score(t) for t in final_particle_set]
final_probs = normalize(exp.(final_scores .- logsumexp(final_scores)))
render_obs_from_particles(final_particle_set, length(final_probs));

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

