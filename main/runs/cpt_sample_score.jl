import JSON

using Circuits
using SpikingCircuits
const Sim = SpikingSimulator
using Distributions: Categorical

includet("../components/value_types.jl")
includet("../components/mux/mux.jl")
includet("../components/ipoisson_gated_repeater.jl")
includet("../components/mux/int_poisson_mux.jl")
includet("../components/cvb.jl")
includet("../components/conditional_sample_score/abstract.jl")
includet("../components/conditional_sample_score/spiking.jl")
includet("../components/thresholded_spike_counter.jl")
includet("../components/to_assmts/abstract.jl")
includet("../components/to_assmts/spiking.jl")
includet("../components/cpt.jl")
includet("../components/cpt_sample_score/abstract.jl")
includet("../components/cpt_sample_score/spiking.jl")

cpt = CPT(
    map(Categorical, [
        [[0.5, 0.5]] [[0.25, 0.75]];
        [[0.9, 0.1]] [[0.1, 0.9]]
    ])
)

unit = SpikingCPTSampleScore(
    CPTSampleScore(cpt, true),
    0.0001,
    2.0
)

circuit = implement_deep(unit, Spiking())
println("Circuit constructed.")