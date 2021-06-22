using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using DiscreteIRTransforms
using Serialization

includet("../../runs/spiketrain_utils.jl")
includet("spiketrain_utils.jl")

# unblocked_events = deserialize("experiments/sprinkler/rundata/mh_unblocked_10000ms_events.jls")
# draw_random_neuron_figure(unblocked_events)

blocked_events = deserialize("experiments/sprinkler/rundata/blocked_mh_6000ms_events.jls")