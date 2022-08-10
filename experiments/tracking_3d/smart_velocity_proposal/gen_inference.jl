using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
import DynamicModels
using ProbEstimates

includet("../model.jl")
includet("../ab_viz.jl")
include("deferred_inference.jl")

ProbEstimates.MinProb() = 0.01
println("MinProb() = $(MinProb())")
# ProbEstimates.use_perfect_weights!()
ProbEstimates.use_noisy_weights!()
ProbEstimates.AssemblySize() = (330)
ProbEstimates.Latency() = (100)
ProbEstimates.UseLowPrecisionMultiply() = false
ProbEstimates.MultAssemblySize() = 200
ProbEstimates.MaxRate() = 0.1 # KHz

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
unweighted_traces_at_each_step_vector = []
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
            push!(unweighted_traces_at_each_step_vector, unweighted_traces_at_each_step)
        end

    # catch
    #     continue
    # end
end
length(final_particle_set)

# animate_pf_results(final_particle_set, tr, true)
# animate_pf_results(final_particle_set, tr, false)
# render_static_trajectories(final_particle_set, tr, true)
# render_static_trajectories(final_particle_set, tr, false)
# final_scores = [get_score(t) for t in final_particle_set]
# final_probs = normalize(exp.(final_scores .- logsumexp(final_scores)))
# render_obs_from_particles(final_particle_set, 100; do_obs=false);


### Spiketrain visualization ###

function surround3(ch, a, dom)
    v = try
        ch[a => :val]
    catch e
        println("ch = ")
        display(ch)
        println("a = $a ; dom = $dom")
        throw(e)
    end
    if v-1 in dom && v+1 in dom
        return (v-1):v+1
    elseif v-1 in dom && v-2 in dom
        return (v-2):v
    else
        return v:(v+2)
    end
end

latent_domains() = (#=vxₜ=Vels(), vyₜ=Vels(), vzₜ=Vels(), =# x=Xs(), y=Ys(), z=Zs(), r=Rs(), true_ϕ=ϕs(), true_θ=θs())
latent_domains_for_viz(ch) = Dict(
        name => surround3(ch, name, dom) for (name, dom) in pairs(latent_domains())
    )

function make_spiketrain_fig(tr, neurons_to_show_indices=1:3; nest_all_at, kwargs...)
    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
    assess_sampling_tree = Dict(
        # :vxₜ => [], :vyₜ => [], :vzₜ => [],
        # :xₜ => [:vxₜ], :yₜ => [:vyₜ], :zₜ => [:vzₜ],
        :x => [], :y => [], :z => [],
        :true_ϕ => [:x, :y, :z],
        :true_θ => [:x, :y, :z],
        :r => [:x, :y, :z, :true_θ, :true_ϕ]
    )

    _propose_sampling_tree = [
        :true_θ => [], :true_ϕ => [],
        :r => [:true_θ, :true_ϕ],
        :x => [:true_θ, :true_ϕ, :r],
        :y => [:true_θ, :true_ϕ, :r],
        :z => [:true_θ, :true_ϕ, :r],
        # :vxₜ => [:true_θ, :true_ϕ, :rₜ],
        # :vyₜ => [:true_θ, :true_ϕ, :rₜ],
        # :vzₜ => [:true_θ, :true_ϕ, :rₜ],
    ]

    propose_addr_topological_order = [p.first for p in _propose_sampling_tree]
    propose_sampling_tree = Dict(_propose_sampling_tree...)
    
    doms = latent_domains_for_viz(get_submap(get_choices(tr), nest_all_at))
    return ProbEstimates.Spiketrains.draw_spiketrain_group_fig(
        ProbEstimates.Spiketrains.value_neuron_scores_groups_noind(keys(doms), values(doms), neurons_to_show_indices), tr,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order);
        nest_all_at, kwargs...
    )
end

f = make_spiketrain_fig(
    last(unweighted_traces_at_each_step_vector[1])[2], 1:10; nest_all_at=(:steps => 1 => :latents),
    resolution=(600, 450), figure_title="Dynamically Weighted Spike Code from Inference"
)



















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

