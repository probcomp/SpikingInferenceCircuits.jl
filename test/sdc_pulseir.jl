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

### Time units: milliseconds

ONRATE = 1.
K = 15

cs = SDCs.ConditionalScore([.5 .5; .5 .5])
pcs = SDCs.PulseConditionalScore(
    ConcreteStreamSamples(
        cs.P,
        30,
        T -> Distributions.Poisson(T * ONRATE)
    ),
    # ΔT, delay, M
    ConcreteAsyncOnGate(30, 0.1, 50,),
    ConcreteThresholdedIndicator(K, 30, 0.1, 50),
    ConcreteOffGate(30, 0.1, 50)
)
# TODO: implement constructor to simplify to something like
# PulseConditionalScore(cs, ΔT, T -> Distributions.Poisson(T * ONRATE), 50, 0.1)

inwindow = PulseIR.Window(
    PulseIR.Interval(0., 0.),
    Inf, Inf
)
obswindow = PulseIR.Window(
    PulseIR.Interval(-1.0, 1.0),
    Inf, Inf
)
println(PulseIR.output_windows(pcs, Dict{Input, PulseIR.Window}(
    (Input(:in_val => i) => inwindow for i=1:2)...,
    (Input(:obs => i) => obswindow for i=1:2)...
)))

with_implemented_subcomponents = SDCs.PulseConditionalScore(
    PoissonStreamSamples(pcs.streamsamples, 1/10_000),
    PoissonAsyncOnGate(pcs.mux_on_gate, 12),
    PoissonThresholdedIndicator(pcs.ti, 12),
    PoissonOffGate(pcs.offgate, 12)
)
println(PulseIR.output_windows(with_implemented_subcomponents, Dict{Input, PulseIR.Window}(
    (Input(:in_val => i) => inwindow for i=1:2)...,
    (Input(:obs => i) => obswindow for i=1:2)...
)))

circuit = implement_deep(with_implemented_subcomponents, Spiking())
nothing

###

# ss_events = SpikingSimulator.simulate_for_time_and_get_events(
#     implement_deep(with_implemented_subcomponents.streamsamples, Spiking()),
#     50;
#     initial_inputs=(1,)
# )

# SpikingSimulator.simulate_for_time(
#     function (itr, time)
#        for (name, evt) in itr
#             println("$time | $evt")
#        end
#     end,
#     #implement_deep(with_implemented_subcomponents.streamsamples, Spiking()),
#     circuit,
#     50;
#     initial_inputs=(:in_val => 1, :obs => 1),
#     # initial_inputs=(1,),
#     event_filter=((compname, event)->event isa SpikingSimulator.Spike)
# )

# indicating_ti_events = SpikingSimulator.simulate_for_time_and_get_events(
#     implement_deep(with_implemented_subcomponents.ti, Spiking()),
#     50;
#     initial_inputs=Tuple(:in for _=1:16)
# )

# off_ti_events = SpikingSimulator.simulate_for_time_and_get_events(
#     implement_deep(with_implemented_subcomponents.ti, Spiking()),
#     50;
#     initial_inputs=Tuple(:in for _=1:14)
# )

# offgate

# ongate

# mux

events = SpikingSimulator.simulate_for_time_and_get_events(circuit, 50;
    initial_inputs=(:in_val => 1, :obs => 1)
)

function spiketrain_dict(event_vector)
    spiketrains = Dict()
    for (time, _, outspike) in event_vector
        if haskey(spiketrains, outspike.name)
            push!(spiketrains[outspike.name], time)
        else
            spiketrains[outspike.name] = [time]
        end
    end
    return spiketrains
end
is_primary_output(compname, event) = (isnothing(compname) && event isa Sim.OutputSpike)
dict = spiketrain_dict(
        filter(((t,args...),) -> is_primary_output(args...), events)
    )

using SpikingCircuits.SpiketrainViz
draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)