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

function simulate_get_output_evts(impl, runtime; 
        inputs)
    inlined, names, state = Circuits.inline(impl)
    inps = [(time, map(x -> Circuits.valname(state.inputs[(x, )]), itr)) 
            for (time, itr) in inputs]
    events = Sim.simulate_for_time_and_get_events(
                                                  inlined,
                                                  runtime;
                                                  inputs=inps
                                                 )
    return [
            (
             t, c,
             typeof(e)(
                       try Circuits.valname(state.outputs[(e.name, )])
                       catch err
                           @error "Couldn't find key $((e.name, )) in $(state.outputs)"
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
