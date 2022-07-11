include("SpiketrainViz.jl")
using .SpiketrainViz

function visualize_labels_spiketrains(label_spiketrain_pairs; kwargs...)
    labels = map(x -> x[1], label_spiketrain_pairs)
    spiketrains = map(x -> x[2], label_spiketrain_pairs)

    return get_spiketrain_figure(spiketrains; names=labels, kwargs...)
end

function visualize_spiketrains_for_labels(labels, events; kwargs...)
    dict = spiketrain_dict(events)
    label_spiketrain_pairs = [
        ("$label", train)
        for label in labels
            for train in [v for (k, v) in dict if k[1] == label && k[2] isa Sim.OutputSpike]
    ]
    visualize_labels_spiketrains(label_spiketrain_pairs; kwargs...)
end


has_prefix(prefix::Pair, pair::Pair) = prefix.first == pair.first && has_prefix(prefix.second, pair.second)
has_prefix(prefix_value, value) = prefix_value == value
has_prefix(prefix_value, pair::Pair) = pair.first == prefix_value
has_prefix(prefix::Pair, value) = false