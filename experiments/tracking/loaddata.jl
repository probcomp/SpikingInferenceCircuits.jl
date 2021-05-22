using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Serialization

include("spiketrain_utils.jl")

smc_events = deserialize("experiments/tracking/rundata/smc_events_2.jls")