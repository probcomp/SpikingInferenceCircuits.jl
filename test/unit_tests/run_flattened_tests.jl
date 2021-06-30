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

function simulate_get_output_evts(impl, runtime; inputs)
    (flattened, in_to_idx, out_to_idx, _) = Circuits.flatten(impl)

    idx_to_out = Any[nothing for _ in out_to_idx]
    for (out, idx) in pairs(out_to_idx)
        idx_to_out[idx] = out
    end

    inps = [(time, map(x -> in_to_idx[x], itr)) for (time, itr) in inputs]
    events = Sim.simulate_for_time_and_get_events(
        flattened,
        runtime;
        inputs=inps
    )
    
    return [
        (
            t, c,
            typeof(e)(
                try idx_to_out[e.name]
                catch err
                    @error "Couldn't find key $(e.name) in $idx_to_out"
                end
            )
        )
        for (t, c, e) in events
            if c === nothing && e isa Sim.OutputSpike
    ]
end

includet("spiketrain_utils.jl")

include("mux.jl")
include("offgate.jl")
include("ti.jl")
include("sync.jl")