using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc

include("model.jl")
using ProbEstimates
# ProbEstimates.use_perfect_weights!()
ProbEstimates.use_noisy_weights!()

model = @DynamicModel(initial_model, step_model, obs_model, 9)
initial_proposal_compiled = @compile_initial_proposal(initial_proposal, 2)
step_proposal_compiled = @compile_step_proposal(step_proposal, 9, 2)
#step_proposal_compiled = @compile_step_proposal(step_model, 9, 2)
#initial_proposal_compiled = @compile_initial_proposal(initial_model, 2)

@load_generated_functions()

NSTEPS = 20
NPARTICLES = 20

tr = simulate(model, (NSTEPS,))


x_traj = [(:steps => i => :latents => :xₜ => :val, X_init + i) for i in 1:NSTEPS]
y_traj = [(:steps => i => :latents => :yₜ => :val, Y_init + i) for i in 1:NSTEPS]
z_traj = [(:steps => i => :latents => :zₜ => :val, Z_init) for i in 1:NSTEPS]
vx_traj = [(:steps => i => :latents => :vxₜ => :val, 1) for i in 1:NSTEPS]
vy_traj = [(:steps => i => :latents => :vyₜ => :val, 1) for i in 1:NSTEPS]
vz_traj = [(:steps => i => :latents => :vzₜ => :val, 0) for i in 1:NSTEPS]

observations = get_dynamic_model_obs(tr)

function surround3(ch, a, dom)
    v = ch[a => :val]
    if v-1 in dom && v+1 in dom
        return (v-1):v+1
    elseif v-1 in dom && v-2 in dom
        return (v-2):v
    else
        return v:(v+2)
    end
end

latent_domains() = (#=vxₜ=Vels(), vyₜ=Vels(), vzₜ=Vels(), =# xₜ=Xs(), yₜ=Ys(), zₜ=Zs(), rₜ=Rs(), exact_ϕ=ϕs(), exact_θ=θs())
latent_domains_for_viz(ch) = Dict(
        name => surround3(ch, name, dom) for (name, dom) in pairs(latent_domains())
    )

function make_spiketrain_fig(tr, neurons_to_show_indices=1:3; nest_all_at, kwargs...)
    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
    assess_sampling_tree = Dict(
        # :vxₜ => [], :vyₜ => [], :vzₜ => [],
        # :xₜ => [:vxₜ], :yₜ => [:vyₜ], :zₜ => [:vzₜ],
        :xₜ => [], :yₜ => [], :zₜ => [],
        :exact_ϕ => [:xₜ, :yₜ, :zₜ],
        :exact_θ => [:xₜ, :yₜ, :zₜ],
        :rₜ => [:xₜ, :yₜ, :zₜ, :exact_θ, :exact_ϕ]
    )

    _propose_sampling_tree = [
        :exact_θ => [], :exact_ϕ => [],
        :rₜ => [:exact_θ, :exact_ϕ],
        :xₜ => [:exact_θ, :exact_ϕ, :rₜ],
        :yₜ => [:exact_θ, :exact_ϕ, :rₜ],
        :zₜ => [:exact_θ, :exact_ϕ, :rₜ],
        # :vxₜ => [:exact_θ, :exact_ϕ, :rₜ],
        # :vyₜ => [:exact_θ, :exact_ϕ, :rₜ],
        # :vzₜ => [:exact_θ, :exact_ϕ, :rₜ],
    ]

    propose_addr_topological_order = [p.first for p in _propose_sampling_tree]
    propose_sampling_tree = Dict(_propose_sampling_tree...)
    
    doms = latent_domains_for_viz(get_submap(get_choices(tr), nest_all_at))
    return ProbEstimates.Spiketrains.draw_spiketrain_group_fig(
        ProbEstimates.Spiketrains.value_neuron_scores_groups(keys(doms), values(doms), neurons_to_show_indices), tr,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order);
        nest_all_at, kwargs...
    )
end

(unweighted_traces_at_each_step, _) = dynamic_model_smc(
    model, observations,
    ch -> (ch[:obs_θ => :val], ch[:obs_ϕ => :val]),
    initial_proposal_compiled, step_proposal_compiled,
    NPARTICLES, # n particles
    ess_threshold=NPARTICLES
)

f = make_spiketrain_fig(
    last(unweighted_traces_at_each_step)[2], 1:3; nest_all_at=(:steps => 1 => :latents),
    resolution=(600, 450), figure_title="Dynamically Weighted Spike Code from Inference"
)