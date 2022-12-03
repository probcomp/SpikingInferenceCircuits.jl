using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
import DynamicModels
using ProbEstimates


# only place this can go wrong is if things are dependent on X_init, Y_init, Z_init and you are making a
# fresh deterministic trace. 



# TODO -- there are some assumptions in the depth calculations that the range of Rs starts at 1. this isn't a
# foregone conclusion if you change X. Fix this. 


# the noise in all model values has been increased -- there is tons of variation now. 
include("../model.jl")
include("../ab_viz.jl")
include("deferred_inference.jl")
include("smart_twostep_proposal.jl")
include("../old/gather_para_trajectories.jl")

println("MinProb() = $(MinProb())")
# ProbEstimates.use_perfect_weights!()
ProbEstimates.use_noisy_weights!()
ProbEstimates.set_assembly_size!(450)
ProbEstimates.set_latency!(20)
ProbEstimates.UseLowPrecisionMultiply() = false
ProbEstimates.MultAssemblySize() = 200
ProbEstimates.MaxRate() = 0.1


step_time = 48
div_time = 1 / (step_time / 1000)
#cmap = make_deterministic_trace()
cmap, norm_xyz  = make_trace_from_realprey(20.833)
#GLMakie.activate!()
#para_3Dtrajectory_in_modelspace(norm_xyz...)
X_init, Y_init, Z_init = norm_xyz[1][1], norm_xyz[2][1], norm_xyz[3][1]
X2, Y2, Z2 = norm_xyz[1][2], norm_xyz[2][2], norm_xyz[3][2]

#NSTEPS = floor(Int64, length(norm_xyz[1]))
NSTEPS = 10
NPARTICLES = 100


model = @DynamicModel(initial_model, step_model, obs_model, 9)
initial_proposal_compiled = @compile_initial_proposal(initial_proposal, 2)
step_proposal_compiled = @compile_step_proposal(step_proposal, 9, 2)
two_timestep_proposal_dumb = @compile_2timestep_proposal(initial_proposal, step_proposal, 9, 2)

@load_generated_functions()

# NSTEPS = 10
# NPARTICLES = 10
# cmap = make_deterministic_trace()

tr, w = generate(model, (NSTEPS,), cmap)
observations = get_dynamic_model_obs(tr);

final_particle_set = []

for i in 1:100
    # try
        (unweighted_traces_at_each_step, weighted_traces) = deferred_dynamic_model_smc(
            model, observations,
            ch -> (ch[:obs_ϕ => :val], ch[:obs_θ => :val]),
       #     two_timestep_proposal_dumb,
            propose_first_two_timesteps_smart,
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
GLMakie.activate!()
animate_pf_results(final_particle_set, tr, true)
animate_pf_results(final_particle_set, tr, false)
render_static_trajectories(final_particle_set, tr, true)
pcoords, gtcoords = render_static_trajectories(final_particle_set, tr, false)
# final_scores = [get_score(t) for t in final_particle_set]
# final_probs = normalize(exp.(final_scores .- logsumexp(final_scores)))
render_obs_from_particles(final_particle_set, 10; do_obs=false);

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

