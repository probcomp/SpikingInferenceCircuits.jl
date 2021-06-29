using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using CPTs
using DiscreteIRTransforms
using Distributions: Normal, cdf

using Serialization

events = deserialize("experiments/better_tracking/rundata/smcrun_may25.jls")

includet("../sprinkler/spiketrain_utils.jl")

draw_random_neuron_figure(events, endtime=1500)