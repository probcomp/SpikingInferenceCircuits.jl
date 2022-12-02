
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