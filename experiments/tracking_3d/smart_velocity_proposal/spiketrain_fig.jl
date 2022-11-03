#=
Util function for selecting relevant neurons to display.
Given a choicemap `ch`, the address `a` of a value,
and its domain `dom`, return the 3 values `ch[a] - 1`, `ch[a]`, and `ch[a] + 1`
=#
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
function get_value(ch, a)
    v = try
        ch[a => :val]
    catch e
        println("ch = ")
        display(ch)
        println("a = $a")
        throw(e)
    end
    return [v]
end

latent_domains() = (#=dx=Vels(), vyₜ=Vels(), vzₜ=Vels(), =#dx=Vels(), x=Xs(), y=Ys(), z=Zs(), r=Rs(), true_ϕ=ϕs(), true_θ=θs())

function latent_domains_for_viz(trs_at_time)
    all_vals = Dict(
        name => dom for (name, dom) in pairs(latent_domains())
    )

    # TODO could add lines for additional values that don't appear in these traces -- e.g. using the `surround3` method above -- if we want
    return all_vals
end

function make_anim_spiketrain_fig(trs_at_each_time, logweights_at_each_time, neurons_to_show_indices=1:10; kwargs...)
    n_particles = length(first(trs_at_each_time))

    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()

    # Hard-code the dependency graph for the `P` and `Q` step generative functions.
    # (We could recover this with static compilation, but I haven't implemented that.)
    assess_sampling_tree = Dict(
        # :dx => [], :vyₜ => [], :vzₜ => [],
        # :xₜ => [:dx], :yₜ => [:vyₜ], :zₜ => [:vzₜ],
        :dx => [],
        :x => [:dx], :y => [], :z => [],
        :true_ϕ => [:x, :y, :z],
        :true_θ => [:x, :y, :z],
        :r => [:x, :y, :z, :true_θ, :true_ϕ]
        # :obs_θ => [:true_θ]
    )
    _propose_sampling_tree = [
        :true_θ => [], :true_ϕ => [],
        :r => [:true_θ, :true_ϕ],
        :x => [:true_θ, :true_ϕ, :r],
        :y => [:true_θ, :true_ϕ, :r],
        :z => [:true_θ, :true_ϕ, :r],
        :dx => [:x],
        # :vyₜ => [:true_θ, :true_ϕ, :rₜ],
        # :vzₜ => [:true_θ, :true_ϕ, :rₜ],
    ]

    propose_addr_topological_order = [p.first for p in _propose_sampling_tree]
    propose_sampling_tree = Dict(_propose_sampling_tree...)

    doms = latent_domains_for_viz(trs_at_each_time)

    # We use this to decide what particle to show spikes from.
    max_weight_idx_at_each_time = [
        findmax(arr)[2] for arr in logweights_at_each_time
    ]

    addr_to_domain = Dict(
        :true_θ => θs(), :true_ϕ => ϕs(), :r => Rs(), :x => Xs(), :y => Ys(), :z => Zs(),
        :dx => Vels(), :obs_θ => θs()
    )

    # We're going to show the P and Q distribution assemblies for these variables,
    # in this order. First element of tuple is a Bool; `true`<> P ; `false`<>Q.
    dists_to_show = [
        (false, :true_θ),
        # (true, :obs_θ),
        (false, :r),
        (false, :x),
        (true, :r),
        (false, :dx),
        (true, :true_θ),
        (true, :x),
        (true, :dx)
    ]
    # We not only need to know what variables to show the sampler neurons for,
    # but what values to show the sampler neurons for.
    dists_vals_to_show = [
        (is_p, addr,
            get_value(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => addr)
        )
        for (is_p, addr) in dists_to_show
    ]

    ### Generate the figure.
    return ProbEstimates.Spiketrains.draw_multiparticle_multistep_spiketrain_group_fig(
        ProbEstimates.Spiketrains.multiparticle_scores_groups(
            keys(doms), values(doms),
            max_weight_idx_at_each_time[1],
            sort(unique(max_weight_idx_at_each_time)),
            dists_vals_to_show,
            neurons_to_show_indices
        ),
        trs_at_each_time, logweights_at_each_time,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order, addr_to_domain);
        timestep_length_to_latency_ratio=5/3,
        figure_title="Spikes from SMC Neurons for 3D Tracking",
        kwargs...
    )
end


function make_anim_spiketrain_fig_and_get_all_L4(trs_at_each_time, logweights_at_each_time, neurons_to_show_indices=1:10; kwargs...)
    n_particles = length(first(trs_at_each_time))

    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()

    # Hard-code the dependency graph for the `P` and `Q` step generative functions.
    # (We could recover this with static compilation, but I haven't implemented that.)
    assess_sampling_tree = Dict(
        # :dx => [], :vyₜ => [], :vzₜ => [],
        # :xₜ => [:dx], :yₜ => [:vyₜ], :zₜ => [:vzₜ],
        :dx => [],
        :x => [:dx], :y => [], :z => [],
        :true_ϕ => [:x, :y, :z],
        :true_θ => [:x, :y, :z],
        :r => [:x, :y, :z, :true_θ, :true_ϕ]
        # :obs_θ => [:true_θ]
    )
    _propose_sampling_tree = [
        :true_θ => [], :true_ϕ => [],
        :r => [:true_θ, :true_ϕ],
        :x => [:true_θ, :true_ϕ, :r],
        :y => [:true_θ, :true_ϕ, :r],
        :z => [:true_θ, :true_ϕ, :r],
        :dx => [:x],
        # :vyₜ => [:true_θ, :true_ϕ, :rₜ],
        # :vzₜ => [:true_θ, :true_ϕ, :rₜ],
    ]

    propose_addr_topological_order = [p.first for p in _propose_sampling_tree]
    propose_sampling_tree = Dict(_propose_sampling_tree...)

    doms = latent_domains_for_viz(trs_at_each_time)

    # We use this to decide what particle to show spikes from.
    max_weight_idx_at_each_time = [
        findmax(arr)[2] for arr in logweights_at_each_time
    ]

    addr_to_domain = Dict(
        :true_θ => θs(), :true_ϕ => ϕs(), :r => Rs(), :x => Xs(), :y => Ys(), :z => Zs(),
        :dx => Vels(), :obs_θ => θs()
    )

    # We're going to show the P and Q distribution assemblies for these variables,
    # in this order. First element of tuple is a Bool; `true`<> P ; `false`<>Q.
    dists_to_show = [
        (false, :true_θ),
        # (true, :obs_θ),
        (false, :r),
        (false, :x),
        (true, :r),
        (false, :dx),
        (true, :true_θ),
        (true, :x),
        (true, :dx)
    ]
    # We not only need to know what variables to show the sampler neurons for,
    # but what values to show the sampler neurons for.
    dists_vals_to_show = [
        (is_p, addr,
            get_value(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => addr)
        )
        for (is_p, addr) in dists_to_show
    ]

    full_specs = [
        ProbEstimates.Spiketrains.LabeledMultiParticleLineGroup(
            ProbEstimates.Spiketrains.FixedText("$(is_p ? "P" : "Q")[addr]"),
            [ProbEstimates.Spiketrains.SubsidiarySingleParticleLineSpec(max_weight_idx_at_each_time[1], ProbEstimates.Spiketrains.DistLine(is_p, addr, v, ProbEstimates.Spiketrains.CountAssembly())) for v in addr_to_domain[addr]]
        )
        for (is_p, addr) in dists_to_show
    ]

    ### Generate the figure.
    (groups_to_show, meta_labels_to_show) = ProbEstimates.Spiketrains.multiparticle_scores_groups(
        keys(doms), values(doms),
        max_weight_idx_at_each_time[1],
        sort(unique(max_weight_idx_at_each_time)),
        dists_vals_to_show,
        neurons_to_show_indices
    )
    return ProbEstimates.Spiketrains.draw_multiparticle_multistep_spiketrain_group_fig_plus_extras(
        (vcat(full_specs, groups_to_show), meta_labels_to_show, length(full_specs)),
        trs_at_each_time, logweights_at_each_time,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order, addr_to_domain);
        timestep_length_to_latency_ratio=5/3,
        figure_title="Spikes from SMC Neurons for 3D Tracking",
        kwargs...
    )
end

### Function for making a different spiketrain visualization.
include("spiketrain_fig_static.jl")