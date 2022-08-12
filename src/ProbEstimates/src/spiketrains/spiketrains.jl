"""
    module Spiketrains

Produce spiketrains from Neural-Gen-Fast inference.

There are 2 types of "specifications" for lines (rows) in a spiketrain figure.
- `SingleParticleLineSpec` : descriptor for a line of spikes (or a line of text) that can occur as a row in a spiketrain,
    given in terms of its relation to a single particle importance-sampling operation
    (e.g. "the spiketrain from the 4th neuron in the P-scoring assembly for variable :x")
- `MultiParticleLineSpec` : descriptor for a line of spikes given in terms of its relation to 
    a multi-particle importance-sampling operation.  (Can include overall particle weight
    and autonormalization line spiketrains, whereas a `SingleParticleLineSpec` cannot.)

These have different interfaces that are used to generate spiketrains; for `SingleParticleLineSpec`s we pass in
a single trace, and for `MultiParticleLineSpec`s we pass in a vector of traces.
"""
module Spiketrains
using Distributions: Exponential, DiscreteUniform
using ProbEstimates: MaxRate, AssemblySize, Latency, K_fwd, K_recip, with_weight_type
using ProbEstimates: ContinuousToDiscreteScoreNumSpikes, LatencyForContinuousToDiscreteScore
import ProbEstimates
using Gen

nest(::Nothing, b) = b
nest(a, b) = a => b
nest(a::Pair, b) = a.first => nest(a.second, b)

abstract type LineSpec end

# spiketrain production for a single importance sample
include("is_spiketrains.jl")
include("single_particle_line_specs.jl")

# spiketrain production for a multi-particle importance sample
# (possibly with auto-normalization for the importance weights)
include("multiparticle_line_specs.jl")
include("autonormalization_spiketrains.jl")

# groups of lines which share a common label
# (e.g. some subset of the lines for neurons in a single assembly)
include("line_groups.jl")

# library of some standard types of spiketrain visualizations
include("library.jl")

# visualization module
include("spiketrain_visualization.jl")

# functions for drawing some default types of spiketrain visualizations,
# which take as input specs of what lines to show (e.g. using the 
# specs in `library.jl`)
include("default_visualizations.jl")

### Exports
export SingleParticleLineSpec, get_line, get_lines, get_label, get_labels
export SampledValue, FwdScoreText, RecipScoreText
export VarValLine, ScoreLine, RecipScoreLine, FwdScoreLine
export CountAssembly, NeuronInCountAssembly, IndLine
export LabeledLineGroup, get_group_labels
export value_neuron_scores_groups, value_neuron_scores_group

export SpiketrainViz
export draw_spiketrain_group_fig

end # module