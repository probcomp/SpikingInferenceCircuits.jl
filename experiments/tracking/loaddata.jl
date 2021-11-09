using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Serialization

smc_events = deserialize("experiments/tracking/rundata/smc_1_events.jls")
