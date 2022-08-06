"""
    module Spiketrains

Produce spiketrains from Neural-Gen-Fast inference.
"""
module Spiketrains
using Distributions: Exponential, DiscreteUniform
using ProbEstimates: MaxRate, AssemblySize, Latency, K_fwd, K_recip, with_weight_type
using ProbEstimates: ContinuousToDiscreteScoreNumSpikes, LatencyForContinuousToDiscreteScore
using Gen

nest(::Nothing, b) = b
nest(a, b) = a => b
nest(a::Pair, b) = a.first => nest(a.second, b)

# spiketrain production for a single importance sample
include("is_spiketrains.jl")
include("line_specs.jl")
include("library.jl") # a couple default spiketrain visualization specs

# spiketrain production for a mutli-particle importance sample
# (possibly with auto-normalization for the importance weights)
include("multiparticle_line_specs.jl")
include("autonormalization_spiketrains.jl")

# visualization module
include("spiketrain_visualization.jl")

# default visualization
function draw_spiketrain_group_fig(groupspecs, tr,
    (prop_sample_tree, assess_sample_tree, prop_addr_top_order);
    resolution=(1280, 720), nest_all_at=nothing, show_lhs_labels=false,
    kwargs...
)
    lines = get_lines(groupspecs, tr,
        (prop_sample_tree, assess_sample_tree, prop_addr_top_order); nest_all_at
    )
    labels = show_lhs_labels ? get_labels(groupspecs) : ["" for _ in lines]
    group_labels = get_group_labels(groupspecs, tr; nest_all_at)
    return SpiketrainViz.draw_spiketrain_figure(lines; labels, group_labels, xmin=0, resolution, kwargs...)
end

### Exports
export LineSpec, get_line, get_lines, get_label, get_labels
export SampledValue, FwdScoreText, RecipScoreText
export VarValLine, ScoreLine, RecipScoreLine, FwdScoreLine
export CountAssembly, NeuronInCountAssembly, IndLine
export LabeledLineGroup, get_group_labels
export value_neuron_scores_groups, value_neuron_scores_group

export SpiketrainViz
export draw_spiketrain_group_fig

end # module