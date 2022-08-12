### "Library" of specs for some standard visualization types ###

function value_neuron_scores_group(a, var_domain, neurons_to_show_indices=1:5;
    name=a,
    show_score_indicators=false,
    particle_idx=nothing,
    show_particle_idx=false
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
        LabeledLineGroup(SampledValue(a, name), [VarValLine(a, v) for v in var_domain]),
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

value_neuron_scores_groups(addrs, var_domains, neurons_to_show_indices=1:5; addr_to_name=identity, kwargs...) =
    (Iterators.flatten(
        value_neuron_scores_group(a, d, neurons_to_show_indices; name=addr_to_name(a), kwargs...)
        for (a, d) in zip(addrs, var_domains)
    ) |> collect)

value_neuron_scores_group_noind(args...; kwargs...) = value_neuron_scores_group(args...; show_score_indicators=false, kwargs...)

function value_neuron_scores_weight_autonorm_groups(
    addrs, var_domains, particle_indices_to_show_vals_scores,
    particle_indices_to_show_weights, neurons_to_show_indices=1:5, kwargs...
)
    val_score_groups = collect(Iterators.flatten([
        value_neuron_scores_groups(addrs, var_domains, neurons_to_show_indices; particle_idx=idx, show_particle_idx=true, kwargs...)
        for idx in particle_indices_to_show_vals_scores
    ]))

    mult_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.MultAssemblySize())
    autonorm_neurons_to_show_indices = 1:min(neurons_to_show_indices[end], ProbEstimates.AutonormalizeRepeaterAssemblysize())
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
    autonorm_group = LabeledMultiParticleLineGroup(FixedText("≈ - log(P[data])"), [
        LogNormalization(NeuronInCountAssembly(i)) for i in autonorm_neurons_to_show_indices
    ])

    return vcat(val_score_groups, weight_groups, [autonorm_group])
end