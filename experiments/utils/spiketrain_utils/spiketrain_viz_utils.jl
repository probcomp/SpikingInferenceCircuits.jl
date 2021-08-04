include("SpiketrainViz.jl")
using .SpiketrainViz

function visualize_labels_spiketrains(label_spiketrain_pairs; kwargs...)
    labels = map(x -> x[1], label_spiketrain_pairs)
    spiketrains = map(x -> x[2], label_spiketrain_pairs)

    return get_spiketrain_figure(spiketrains; names=labels, kwargs...)
end