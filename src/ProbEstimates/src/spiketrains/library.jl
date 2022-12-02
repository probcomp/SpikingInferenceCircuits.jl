### "Library" of specs for some standard visualization types ###

function value_neuron_scores_group(a, var_domain, neurons_to_show_indices=1:5;
    addr_to_name=identity,
    name=addr_to_name(a),
    show_score_indicators=false,
    particle_idx=nothing,
    show_particle_idx=false,
    include_values=true
)
    LabeledLineGroup = (
        if isnothing(particle_idx)
            LabeledSingleParticleLineGroup
        else
            (labelspec, linespecs) -> LabeledMultiParticleLineGroup(
                SingleParticleTextWrapper(particle_idx, labelspec, show_particle_idx),
                [SubsidiarySingleParticleLineSpec(particle_idx, spec) for spec in linespecs]
            )
        end
    )

    return [
        (include_values ? [
            LabeledLineGroup(SampledValue(a, name), [VarValLine(a, v) for v in var_domain])
        ] : [])...,
        LabeledLineGroup(RecipScoreText(a, name), [
            [RecipScoreLine(a, NeuronInCountAssembly(i)) for i in neurons_to_show_indices]...,
            (show_score_indicators ? [RecipScoreLine(a, IndLine())] : [])...
        ]),
        LabeledLineGroup(FwdScoreText(a, name), [
            [FwdScoreLine(a, NeuronInCountAssembly(i)) for i in neurons_to_show_indices]...,
            (show_score_indicators ? [FwdScoreLine(a, IndLine())] : [])...
        ]),
    ]
end

function value_neuron_scores_groups(addrs, var_domains, neurons_to_show_indices=1:5; addr_to_name=identity, flatten=true, kwargs...)
    println("adddrs = $addrs ; addr_to_name.(addrs) = $(addr_to_name.(addrs))")
    itr = (
        value_neuron_scores_group(a, d, neurons_to_show_indices; name=addr_to_name(a), kwargs...)
        for (a, d) in zip(addrs, var_domains)
    )
    if flatten
        return collect(Iterators.flatten(itr))
    else
        return collect(itr)
    end
end
scores_groups(args...; kwargs...) =
    value_neuron_scores_groups(args...; include_values=false, kwargs...)

function get_dist_groups_for_one_var(addr, vals, neurons_to_show_indices, GroupConstructor; addr_to_name=identity, name=addr_to_name(addr), is_p, val_to_label=identity)
    if is_p
        p_or_q = "P"
    else
        p_or_q = "Q"
    end
    return [
        GroupConstructor(
            "$p_or_q[$name = $(val_to_label(v))]",
            [DistLine(is_p, addr, v, NeuronInCountAssembly(i)) for i in neurons_to_show_indices]
        )
        for v in vals
    ]
end

# function get_dist_groups(
#     addrs, variables_vals_to_show_p_dists_for,
#     variables_vals_to_show_q_dists_for, neurons_to_show_indices;
#     particle_idx, show_particle_idx, addr_to_name,
#     val_to_label=identity,
#     kwargs...
# )
#     @assert !isnothing(particle_idx) "DistGroups currently only implemented for MultiParticleLineGroup, since it used FixedText.  [this reflects some bad design]"
#     GroupConstructor = (
#         (label, linespecs) -> LabeledMultiParticleLineGroup(
#             FixedText(show_particle_idx ? "P$particle_idx:$label" : label),
#             [SubsidiarySingleParticleLineSpec(particle_idx, spec) for spec in linespecs]
#         )
#     )

#     return vcat(
#         collect(Iterators.flatten(
#             get_dist_groups_for_one_var(addr, vals, neurons_to_show_indices, GroupConstructor; addr_to_name, val_to_label, is_p=true)
#             for (addr, vals) in variables_vals_to_show_p_dists_for
#         )),
#         collect(Iterators.flatten(
#             get_dist_groups_for_one_var(addr, vals, neurons_to_show_indices, GroupConstructor; addr_to_name, val_to_label, is_p=false)
#             for (addr, vals) in variables_vals_to_show_q_dists_for
#         ))
#     )
# end

# Input is a list of (is_p::Bool, addr::Symbol, vals::Iterator)
function get_dist_groups(
    addrs,
    variables_vals_type_to_show_dists_for, neurons_to_show_indices;
    particle_idx, show_particle_idx, addr_to_name,
    val_to_label=identity,
    kwargs...
)
    @assert !isnothing(particle_idx) "DistGroups currently only implemented for MultiParticleLineGroup, since it used FixedText.  [this reflects some bad design]"
    GroupConstructor = (
        (label, linespecs) -> LabeledMultiParticleLineGroup(
            FixedText(show_particle_idx ? "P$particle_idx:$label" : label),
            [SubsidiarySingleParticleLineSpec(particle_idx, spec) for spec in linespecs]
        )
    )

    return collect(Iterators.flatten(
        get_dist_groups_for_one_var(addr, vals, neurons_to_show_indices, GroupConstructor; addr_to_name, val_to_label, is_p)
        for (is_p, addr, vals) in variables_vals_type_to_show_dists_for
    ))
end

value_neuron_scores_group_noind(args...; kwargs...) = value_neuron_scores_group(args...; show_score_indicators=false, kwargs...)

function value_neuron_scores_weight_autonorm_groups_grouped_by_variable(
    addrs, var_domains, particle_indices_to_show_vals_scores,
    particle_indices_to_show_weights, neurons_to_show_indices=1:5;
    mult_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.MultAssemblySize()),
    autonorm_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.AutonormalizeRepeaterAssemblysize()),
    kwargs...
)
    val_score_groups = collect(Iterators.flatten([
        value_neuron_scores_groups(addrs, var_domains, neurons_to_show_indices; particle_idx=idx, show_particle_idx=true, kwargs...)
        for idx in particle_indices_to_show_vals_scores
    ]))

    weight_groups = [
        LabeledMultiParticleLineGroup(
            FixedText("Particle $part_idx normalized weight"),
            [NormalizedWeight(part_idx, NeuronInCountAssembly(neuron_idx)) for neuron_idx in mult_neurons_to_show_indices]
        )
        for part_idx in particle_indices_to_show_weights
    ]
    
    # LabeledMultiParticleLineGroup(
    #     FixedText("Particle weights"),
    #     collect(Iterators.flatten([
    #         [NormalizedWeight(part_idx, NeuronInCountAssembly(neuron_idx)) for neuron_idx in mult_neurons_to_show_indices]
    #         for part_idx in particle_indices_to_show_weights
    #     ]))
    # )
    autonorm_group = LabeledMultiParticleLineGroup(FixedText("≈-log(P[d])"), [
        LogNormalization(NeuronInCountAssembly(i)) for i in autonorm_neurons_to_show_indices
    ])

    return vcat(val_score_groups, weight_groups, [autonorm_group])
end

function value_neuron_scores_weight_autonorm_groups(
    addrs, var_domains, particle_indices_to_show_vals_scores,
    particle_indices_to_show_weights, neurons_to_show_indices=1:5;
    mult_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.MultAssemblySize()),
    autonorm_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.AutonormalizeRepeaterAssemblysize()),
    addr_to_name=identity,
    kwargs...
)
    println("addr_to_name = $addr_to_name")
    val_score_groups = collect(Iterators.flatten(
        value_neuron_scores_groups(addrs, var_domains, neurons_to_show_indices; particle_idx=idx, show_particle_idx=true, flatten=false, addr_to_name, kwargs...)
        for idx in particle_indices_to_show_vals_scores
    ))
    @assert all(length(x) == 3 for x in val_score_groups)
    val_groups = map(x -> x[1], val_score_groups)
    q_groups = map(x -> x[2], val_score_groups)
    p_groups = map(x -> x[3], val_score_groups)

    weight_groups = [
        LabeledMultiParticleLineGroup(
            FixedText("P$part_idx: w̃"),
            [NormalizedWeight(part_idx, NeuronInCountAssembly(neuron_idx)) for neuron_idx in mult_neurons_to_show_indices]
        )
        for part_idx in particle_indices_to_show_weights
    ]
    autonorm_group = LabeledMultiParticleLineGroup(FixedText("≈-log(P[d])"), [
        LogNormalization(NeuronInCountAssembly(i)) for i in autonorm_neurons_to_show_indices
    ])

    score_groups = collect(Iterators.flatten(zip(q_groups, p_groups)))

    return (
        vcat(val_groups, score_groups, weight_groups, [autonorm_group]), # groups
        [ # layer labels
            ("Layer 2/3", length(val_groups)),
            ("Layer 5/6", length(score_groups) + length(weight_groups) + 1),
            # ("W", length(weight_groups)),
            # ("AN", length([autonorm_group]))
        ]
    )
end
sum_of_lengths(arr) = sum(map(x -> x[2], arr))

function value_neuron_scores_dists_weight_autonorm_groups(
    addrs, var_domains, particle_indices_to_show_vals_scores,
    particle_indices_to_show_weights,
    variables_vals_to_show_p_dists_for,
    variables_vals_to_show_q_dists_for, neurons_to_show_indices=1:5;
    mult_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.MultAssemblySize()),
    autonorm_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.AutonormalizeRepeaterAssemblysize()),
    addr_to_name=identity, val_to_label=identity,
    kwargs...
)
    # println("addr_to_name = $addr_to_name")
    val_score_groups = collect(Iterators.flatten(
        value_neuron_scores_groups(addrs, var_domains, neurons_to_show_indices; particle_idx=idx, show_particle_idx=true, flatten=false, addr_to_name, kwargs...)
        for idx in particle_indices_to_show_vals_scores
    ))
    @assert all(length(x) == 3 for x in val_score_groups)
    val_groups = map(x -> x[1], val_score_groups)
    q_groups = map(x -> x[2], val_score_groups)
    p_groups = map(x -> x[3], val_score_groups)

    dist_groups = collect(Iterators.flatten(
        get_dist_groups(addrs, variables_vals_to_show_p_dists_for, variables_vals_to_show_q_dists_for, neurons_to_show_indices;
            particle_idx=idx, show_particle_idx=true, flatten=false, addr_to_name, val_to_label, kwargs...
        )
        for idx in particle_indices_to_show_vals_scores
    ))

    weight_groups = [
        LabeledMultiParticleLineGroup(
            FixedText("P$part_idx: w̃"),
            [NormalizedWeight(part_idx, NeuronInCountAssembly(neuron_idx)) for neuron_idx in mult_neurons_to_show_indices]
        )
        for part_idx in particle_indices_to_show_weights
    ]
    autonorm_group = LabeledMultiParticleLineGroup(FixedText("≈-log(P[d])"), [
        LogNormalization(NeuronInCountAssembly(i)) for i in autonorm_neurons_to_show_indices
    ])

    score_groups = collect(Iterators.flatten(zip(q_groups, p_groups)))

    return (
        vcat(val_groups, dist_groups, score_groups, weight_groups, [autonorm_group]), # groups
        [ # layer labels
            ("Layer 2/3", length(val_groups)),
            ("Layer 4", length(dist_groups)),
            ("Layer 5/6", length(score_groups) + length(weight_groups) + 1),
            # ("W", length(weight_groups)),
            # ("AN", length([autonorm_group]))
        ]
    )
end

function multiparticle_scores_groups( # TODO: rename to sampler_groups
    addrs, var_domains, particle_indices_to_show_vals_scores,
    particle_indices_to_show_weights,
    variables_vals_to_show_dists_for, neurons_to_show_indices=1:5;
    mult_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.MultAssemblySize()),
    autonorm_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.AutonormalizeRepeaterAssemblysize()),
    addr_to_name=identity, val_to_label=identity,
    kwargs...
)

    dist_groups = collect(Iterators.flatten(
        get_dist_groups(addrs, variables_vals_to_show_dists_for, neurons_to_show_indices;
            particle_idx=idx, show_particle_idx=true, flatten=false, addr_to_name, val_to_label, kwargs...
        )
        for idx in particle_indices_to_show_vals_scores
    ))

    # LabeledLineGroup(
    #     [
    #         DistLineSpec(true, :true_θ, -0.4, NeuronInAssembly(4))
    #     ]
    # )

    return (
        # score_groups,
        dist_groups, # score_groups
        [ # layer labels
            ("Layer 4", length(dist_groups)),
            # ("Layer 5/6", length(score_groups)),
            # ("W", length(weight_groups)),
            # ("AN", length([autonorm_group]))
        ]
    )
end

### TODO: fix calls to old `get_dist_groups` interface