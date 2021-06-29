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
    0., 2.
)

includet("../spiketrain_utils.jl")

events = Sim.simulate_for_time_and_get_events(
    implement_deep(og, Spiking()),
    10,
    inputs=[
        (0., (:on,)),
        (1., (:in,)),
        (2.5, (:in,))
    ]
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        (compname === :ss || compname === nothing) && event isa Sim.OutputSpike
    end
        # filter(((t,args...),) -> is_primary_output(args...), events)
    )
