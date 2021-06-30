using Test
using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
const Sim = SpikingCircuits.SpikingSimulator
import Distributions

using .PulseIR: ConcreteStreamSamples, ConcreteThresholdedIndicator, ConcreteOffGate, ConcreteAsyncOnGate
using .PulseIR: PoissonStreamSamples, PoissonThresholdedIndicator, PoissonOffGate, PoissonAsyncOnGate
using .SDCs: PulseMux
includet("spiketrain_utils.jl")

function simulate_get_output_evts(impl, runtime; inputs)
    events = Sim.simulate_for_time_and_get_events(
        impl,
        runtime;
        inputs
    )
    return [
        (t, c, e)
        for (t, c, e) in events
            if c === nothing && e isa Sim.OutputSpike
    ]
end

include("mux.jl")
include("offgate.jl")
include("ti.jl")
include("sync.jl")