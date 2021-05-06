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

og = PoissonAsyncOnGate(
    ConcreteAsyncOnGate(300, 0.1, 50),
    30
)
# mux = SDCs.PulseMux(SDCs.Mux(2, SpikeWire()), og)

# imp = implement_deep(mux, Spiking())

# events = Sim.simulate_for_time_and_get_events(
#     imp, 10, initial_inputs=(:sel => 1,)
# )
# dict = spiketrain_dict(
#     filter(events) do (t, compname, event)
#         (compname === :ss || compname === nothing) && event isa Sim.OutputSpike
#     end
#         # filter(((t,args...),) -> is_primary_output(args...), events)
#     )

events = Sim.simulate_for_time_and_get_events(
    implement_deep(og, Spiking()),
    10, initial_inputs=(:on,)
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        (compname === :ss || compname === nothing) && event isa Sim.OutputSpike
    end
        # filter(((t,args...),) -> is_primary_output(args...), events)
    )
