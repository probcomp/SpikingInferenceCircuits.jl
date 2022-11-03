function draw_singleparticle_spiketrain_group_fig(groupspecs, tr,
    (prop_sample_tree, assess_sample_tree, prop_addr_top_order);
    resolution=(1280, 720), nest_all_at=nothing, show_lhs_labels=false,
    kwargs...
)
    lines = get_lines(groupspecs, tr,
        (prop_sample_tree, assess_sample_tree, prop_addr_top_order); nest_all_at
    )
    labels = show_lhs_labels ? get_labels(groupspecs) : ["" for _ in lines]
    group_labels = get_group_labels(groupspecs, tr; nest_all_at)
    colors = SpiketrainViz.get_colors(groupspecs)
    return SpiketrainViz.draw_spiketrain_figure(lines; labels, group_labels, xmin=0, resolution, colors, kwargs...)
end

# default visualization for multi-trace visualizations
function draw_multiparticle_spiketrain_group_fig(groupspecs, trs, log_trace_weights,
    (prop_sample_tree, assess_sample_tree, prop_addr_top_order);
    resolution=(1280, 720), nest_all_at=nothing,
    kwargs...
)
    lines = get_lines_for_multiparticle_spec_groups(
        groupspecs, trs, 
        log_trace_weights,
        (propose_sampling_tree, assess_sampling_tree, propose_topological_order)
    )
    group_labels = get_group_labels_for_multiparticle_specs(groupspecs, trs; nest_all_at)
    colors = SpiketrainViz.get_colors(groupspecs)
    return SpiketrainViz.draw_spiketrain_figure(lines; labels, group_labels, xmin=0, resolution, colors, kwargs...)
end

default_t_to_nesting_address(t) =
    if t == 0
        :init => :latents
    else
        :steps => t => :latents
    end

# Now `trs`, `log_trace_weights` are indexed by `[time][particle_index]`.
function draw_multiparticle_multistep_spiketrain_group_fig(
    (groupspecs, layer_labels_and_group_lengths), trs, log_trace_weights,
    (prop_sample_tree, assess_sample_tree, prop_addr_top_order, addr_to_domain);
    resolution=(1280, 720), time_to_nesting_addr=default_t_to_nesting_address,
    # Default: ~1.1 for weight readout, .1 for autonormalization excitatory spikes, 1 for weight readout
    timestep_length_to_latency_ratio=2.5,
    return_metadata=false,
    show_cortex_layer_groups=true,
    first_label_length=118,
    addr_to_name=identity,
    kwargs...
)
    ## Get lines over multiple timesteps
    ms_per_timestep = timestep_length_to_latency_ratio * Latency()
    lines = []
    starttime = 0
    for (t_plus_1, (trs_at_step, logweights_at_step)) in enumerate(zip(trs, log_trace_weights))
        lines_now = get_lines_for_multiparticle_spec_groups(
            groupspecs, trs_at_step, 
            logweights_at_step,
            (prop_sample_tree, assess_sample_tree, prop_addr_top_order, addr_to_domain);
            nest_all_at=time_to_nesting_addr(t_plus_1 - 1)
        )
        if isempty(lines)
            lines = copy(lines_now)
        else
            @assert length(lines) == length(lines_now)
            for i in eachindex(lines)
                append!(lines[i], lines_now[i] .+ starttime)
            end
        end

        starttime += ms_per_timestep
    end
    
    # TODO: group labels compatible with multi-timestep
    # for now, placeholder:
    # group_labels = [(repr(g.label_spec), length(g.line_specs)) for g in groupspecs]

    group_labels_and_lengths = get_static_group_labels_for_multiparticle_spec_groups(groupspecs; addr_to_name)
    ctr = 1
    layer_label_and_full_lengths = []
    for (label, len) in layer_labels_and_group_lengths
        push!(layer_label_and_full_lengths, (label, sum(l for (_, l) in group_labels_and_lengths[ctr:ctr + len - 1])))
        ctr += len
    end

    if show_cortex_layer_groups
        group_labels = [
            (group_labels_and_lengths, 0),
            (layer_label_and_full_lengths, first_label_length)
        ]
    else
        group_labels = [(group_labels_and_lengths, 0)]
    end

    colors = SpiketrainViz.get_colors(groupspecs)
    viz = SpiketrainViz.draw_spiketrain_figure_animated(lines; group_labels, xmin=0, resolution, colors, kwargs...)

    if return_metadata
        return (viz, (lines, group_labels, colors))
    else
        return viz
    end
end

function draw_multiparticle_multistep_spiketrain_group_fig_plus_extras(
    (groupspecs, layer_labels_and_group_lengths, n_hidden_groups), trs, log_trace_weights,
    (prop_sample_tree, assess_sample_tree, prop_addr_top_order, addr_to_domain);
    resolution=(1280, 720), time_to_nesting_addr=default_t_to_nesting_address,
    # Default: ~1.1 for weight readout, .1 for autonormalization excitatory spikes, 1 for weight readout
    timestep_length_to_latency_ratio=2.5,
    return_metadata=false,
    show_cortex_layer_groups=true,
    first_label_length=118,
    addr_to_name=identity,
    kwargs...
)
    ## Get lines over multiple timesteps
    ms_per_timestep = timestep_length_to_latency_ratio * Latency()
    lines = []
    starttime = 0
    for (t_plus_1, (trs_at_step, logweights_at_step)) in enumerate(zip(trs, log_trace_weights))
        lines_now = get_lines_for_multiparticle_spec_groups(
            groupspecs, trs_at_step, 
            logweights_at_step,
            (prop_sample_tree, assess_sample_tree, prop_addr_top_order, addr_to_domain);
            nest_all_at=time_to_nesting_addr(t_plus_1 - 1)
        )
        if isempty(lines)
            lines = copy(lines_now)
        else
            @assert length(lines) == length(lines_now)
            for i in eachindex(lines)
                append!(lines[i], lines_now[i] .+ starttime)
            end
        end

        starttime += ms_per_timestep
    end
    
    # TODO: group labels compatible with multi-timestep
    # for now, placeholder:
    # group_labels = [(repr(g.label_spec), length(g.line_specs)) for g in groupspecs]

    group_labels_and_lengths = get_static_group_labels_for_multiparticle_spec_groups(groupspecs; addr_to_name)

    visible_group_specs = groupspecs[n_hidden_groups+1:end]
    visible_lines = lines[sum([len for (_, len) in group_labels_and_lengths[1:n_hidden_groups]])+1:end]
    visible_group_labels_and_lengths = group_labels_and_lengths[n_hidden_groups+1:end]

    ctr = 1
    layer_label_and_full_lengths = []
    for (label, len) in layer_labels_and_group_lengths
        push!(layer_label_and_full_lengths, (label, sum(l for (_, l) in visible_group_labels_and_lengths[ctr:ctr + len - 1])))
        ctr += len
    end

    if show_cortex_layer_groups
        visible_group_labels = [
            (visible_group_labels_and_lengths, 0),
            (layer_label_and_full_lengths, first_label_length)
        ]
    else
        visible_group_labels = [(visible_group_labels_and_lengths, 0)]
    end


    colors = SpiketrainViz.get_colors(visible_group_specs)
    viz = SpiketrainViz.draw_spiketrain_figure_animated(visible_lines; group_labels=visible_group_labels, xmin=0, resolution, colors, kwargs...)

    if return_metadata
        return (viz, (lines, visible_group_labels, colors, length(lines) - length(visible_lines)))
    else
        return viz
    end
end