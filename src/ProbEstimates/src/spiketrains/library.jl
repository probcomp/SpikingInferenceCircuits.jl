### "Library" of specs for some standard visualization types ###

value_neuron_scores_group(a, var_domain, neurons_to_show_indices=1:5; name=a) = [
    LabeledLineGroup(SampledValue(a, name), [VarValLine(a, v) for v in var_domain]),
    LabeledLineGroup(RecipScoreText(a, name), [
        [RecipScoreLine(a, NeuronInCountAssembly(i)) for i in neurons_to_show_indices]...,
        RecipScoreLine(a, IndLine())
    ]),
    LabeledLineGroup(FwdScoreText(a, name), [
        [FwdScoreLine(a, NeuronInCountAssembly(i)) for i in neurons_to_show_indices]...,
        FwdScoreLine(a, IndLine())
    ]),
]
value_neuron_scores_groups(addrs, var_domains, neurons_to_show_indices=1:5; addr_to_name=identity) =
    (Iterators.flatten(
        value_neuron_scores_group(a, d, neurons_to_show_indices; name=addr_to_name(a))
        for (a, d) in zip(addrs, var_domains)
    ) |> collect) :: Vector{LabeledLineGroup}

value_neuron_scores_group_noind(a, var_domain, neurons_to_show_indices=1:5; name=a) = [
    LabeledLineGroup(SampledValue(a, name), [VarValLine(a, v) for v in var_domain]),
    LabeledLineGroup(RecipScoreText(a, name), [
        [RecipScoreLine(a, NeuronInCountAssembly(i)) for i in neurons_to_show_indices]...,
    ]),
    LabeledLineGroup(FwdScoreText(a, name), [
        [FwdScoreLine(a, NeuronInCountAssembly(i)) for i in neurons_to_show_indices]...,
    ]),
]
value_neuron_scores_groups_noind(addrs, var_domains, neurons_to_show_indices=1:5; addr_to_name=identity) =
    (Iterators.flatten(
        value_neuron_scores_group_noind(a, d, neurons_to_show_indices; name=addr_to_name(a))
        for (a, d) in zip(addrs, var_domains)
    ) |> collect) :: Vector{LabeledLineGroup}