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
function surround1(ch, a, dom)
    v = try
        ch[a => :val]
    catch e
        println("ch = ")
        display(ch)
        println("a = $a ; dom = $dom")
        throw(e)
    end
    return [v]
end

latent_domains() = (#=dx=Vels(), vyₜ=Vels(), vzₜ=Vels(), =# x=Xs(), y=Ys(), z=Zs(), r=Rs(), true_ϕ=ϕs(), true_θ=θs())

function latent_domains_for_viz(trs_at_time)
    # nesting_addrs = [ProbEstimates.Spiketrains.default_t_to_nesting_address(tp1 - 1) for tp1=1:length(trs_at_time)]
    # Note that here we are hardcoding that we're only putting in the values for trace 1.
    # This seems like something that might change.
    # all_vals = Dict(
    #     name => [get_submap(get_choices(trs[1]), nesting_addr)[name => :val] for (trs, nesting_addr) in zip(trs_at_time, nesting_addrs)]
    #     for name in keys(latent_domains())
    # )

    all_vals = Dict(
        name => dom for (name, dom) in pairs(latent_domains())
    )

    # TODO could add lines for additional values that don't appear in these traces -- e.g. using the `surround3` method above -- if we want
    return all_vals
end

function make_spiketrain_fig(trs_at_each_time, logweights_at_each_time, neurons_to_show_indices=1:10; kwargs...)
    n_particles = length(first(trs_at_each_time))

    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
    assess_sampling_tree = Dict(
        # :dx => [], :vyₜ => [], :vzₜ => [],
        # :xₜ => [:dx], :yₜ => [:vyₜ], :zₜ => [:vzₜ],
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
        # :dx => [:true_θ, :true_ϕ, :rₜ],
        # :vyₜ => [:true_θ, :true_ϕ, :rₜ],
        # :vzₜ => [:true_θ, :true_ϕ, :rₜ],
    ]

    propose_addr_topological_order = [p.first for p in _propose_sampling_tree]
    propose_sampling_tree = Dict(_propose_sampling_tree...)

    doms = latent_domains_for_viz(trs_at_each_time)

    max_weight_idx_at_each_time = [
        findmax(arr)[2] for arr in logweights_at_each_time
    ]

    addr_to_domain = Dict(
        :true_θ => θs(), :true_ϕ => ϕs(), :r => Rs(), :x => Xs(), :y => Ys(), :z => Zs()
    )

    variables_vals_to_show_p_dists_for = [
        (:x, surround3(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => :x, Xs()))
    ]
    variables_vals_to_show_q_dists_for = [
        (:x, surround3(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => :x, Xs()))
    ]

    return ProbEstimates.Spiketrains.draw_multiparticle_multistep_spiketrain_group_fig(
        ProbEstimates.Spiketrains.value_neuron_scores_dists_weight_autonorm_groups(
            keys(doms), values(doms),
            max_weight_idx_at_each_time[1],
            sort(unique(max_weight_idx_at_each_time)),
            variables_vals_to_show_p_dists_for,
            variables_vals_to_show_q_dists_for,
            neurons_to_show_indices
        ),
        trs_at_each_time, logweights_at_each_time,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order, addr_to_domain);
        timestep_length_to_latency_ratio=5/3,
        figure_title="Spikes from SMC Neurons for 3D Tracking",
        kwargs...
    )
end

function make_anim_spiketrain_fig(trs_at_each_time, logweights_at_each_time, neurons_to_show_indices=1:10; kwargs...)
    n_particles = length(first(trs_at_each_time))

    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
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

    max_weight_idx_at_each_time = [
        findmax(arr)[2] for arr in logweights_at_each_time
    ]

    addr_to_domain = Dict(
        :true_θ => θs(), :true_ϕ => ϕs(), :r => Rs(), :x => Xs(), :y => Ys(), :z => Zs(),
        :dx => Vels(), :obs_θ => θs()
    )

    # variables_vals_to_show_p_dists_for = [
        # (:x, surround3(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => :x, Xs()))
    # ]
    # variables_vals_to_show_q_dists_for = [
    #     (:x, surround3(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => :x, Xs()))
    # ]

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
    dists_vals_to_show = [
        (is_p, addr,
            surround1(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => addr, addr_to_domain[addr])
        )
        for (is_p, addr) in dists_to_show
    ]
    println("vals to show: $dists_vals_to_show")

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

# function get_groupspecs(trs_at_each_time, logweights_at_each_time, neurons_to_show_indices=1:10; kwargs...)
#     n_particles = length(first(trs_at_each_time))

#     ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
#     assess_sampling_tree = Dict(
#         # :dx => [], :vyₜ => [], :vzₜ => [],
#         # :xₜ => [:dx], :yₜ => [:vyₜ], :zₜ => [:vzₜ],
#         :x => [], :y => [], :z => [],
#         :true_ϕ => [:x, :y, :z],
#         :true_θ => [:x, :y, :z],
#         :r => [:x, :y, :z, :true_θ, :true_ϕ]
#     )

#     _propose_sampling_tree = [
#         :true_θ => [], :true_ϕ => [],
#         :r => [:true_θ, :true_ϕ],
#         :x => [:true_θ, :true_ϕ, :r],
#         :y => [:true_θ, :true_ϕ, :r],
#         :z => [:true_θ, :true_ϕ, :r],
#         # :dx => [:true_θ, :true_ϕ, :rₜ],
#         # :vyₜ => [:true_θ, :true_ϕ, :rₜ],
#         # :vzₜ => [:true_θ, :true_ϕ, :rₜ],
#     ]

#     propose_addr_topological_order = [p.first for p in _propose_sampling_tree]
#     propose_sampling_tree = Dict(_propose_sampling_tree...)

#     doms = latent_domains_for_viz(trs_at_each_time)

#     ProbEstimates.Spiketrains.value_neuron_scores_weight_autonorm_groups(
#         keys(doms), values(doms), 1, 1:n_particles, neurons_to_show_indices
#     )
# end