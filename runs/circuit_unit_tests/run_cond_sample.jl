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

ONRATE = 0.1 # spike/ms
K = 5

cs = SDCs.ConditionalSample([.5 .5; .5 .5])
pcs = SDCs.PulseConditionalSample(
    ConcreteStreamSamples(
        cs.P,
        300,
        T -> Distributions.Poisson(T * ONRATE)
    ),
    # ΔT, delay, M
    ConcreteAsyncOnGate(300, 0.1, 50,),
    ConcreteOffGate(300, 0.1, 50),
    # The K + 1 here is important!
    ConcreteThresholdedIndicator(K + 1, 300, 0.1, 50),
    ConcreteOffGate(300, 0.1, 50),
    50.0, .1
)

inwindow = PulseIR.Window(
    PulseIR.Interval(0., 0.),
    Inf, Inf
)
println(PulseIR.output_windows(pcs, Dict{Input, PulseIR.Window}(
    Input(:in_val => i) => inwindow for i=1:2
)))

# Want the TI to have a fast rate, so we DONT output the spike which turns
# off the gate.
# TODO: can we move this stuff about K vs K+1, and the rates, to the ProbCounter file?
with_implemented_subcomponents = SDCs.PulseConditionalSample(
    PoissonStreamSamples(pcs.streamsamples, 1/(10^10)),
    PoissonAsyncOnGate(pcs.mux_on_gate, 12),
    PoissonOffGate(pcs.wta_offgate, 12),
    PoissonThresholdedIndicator(pcs.ti, 30),
    PoissonOffGate(pcs.offgate, 12),
    50.0, .1
)
println(PulseIR.output_windows(with_implemented_subcomponents, Dict{Input, PulseIR.Window}(
    Input(:in_val => i) => inwindow for i=1:2
)))

circuit = implement_deep(with_implemented_subcomponents, Spiking())

events = SpikingSimulator.simulate_for_time_and_get_events(circuit, 500;
    initial_inputs=(:in_val => 1,)
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
    filter(events) do (t, compname, event)
        (compname === :ss || compname === nothing) && event isa Sim.OutputSpike
    end
        # filter(((t,args...),) -> is_primary_output(args...), events)
    )

using SpikingCircuits.SpiketrainViz
draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)

### Below here are some functions, etc., to test that the outputted inverse rates look correct

draw_strain(dict) = draw_spiketrain_figure( collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0)

event_vecs = [
    SpikingSimulator.simulate_for_time_and_get_events(circuit, 500;
        initial_inputs=(:in_val => 1,)
    )
    for _=1:100
]

dicts = [
    spiketrain_dict(
        filter(ev) do (t, compname, event)
            (compname === :ss || compname === nothing) && event isa Sim.OutputSpike
        end
    ) for ev in event_vecs
]

lens = [
    length(dict[:inverse_prob]) for dict in dicts
]
maxtimes = [
    maximum(dict[:inverse_prob]) for dict in dicts
]
sum(lens)

# function manual_count(dict::Dict)
#     i1 = 1
#     i2 = 1
#     total = 0
#     cnt = 0
#     sample = haskey(dict, :value => 1) ? 1 : 2
#     while cnt < K
#         i = argmin([dict[1][i1], dict[2][i2]])
#         if i == 1
#             i1 += 1
#         else
#             i2 += 1
#         end
#         if sample == i
#             cnt += 1
#         end
#         total += 1
#     end
#     return total
# end

# manually_count(dicts::Vector{<:Dict}) = map(manual_count, dicts)

# mcnts = manually_count(dicts)