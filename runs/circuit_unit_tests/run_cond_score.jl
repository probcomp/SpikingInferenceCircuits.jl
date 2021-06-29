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
    PoissonStreamSamples(pcs.streamsamples, 10^(-10)),
    PoissonAsyncOnGate(pcs.mux_on_gate, 30),
    # Note how there is a slower rate on the TI.  I _think_ this is important for scoring precisely,
    # so that spikes will pass through the output before turning off the unit.
    PoissonThresholdedIndicator(pcs.ti, 10),
    PoissonOffGate(pcs.offgate, 30)
)
println(PulseIR.output_windows(with_implemented_subcomponents, Dict{Input, PulseIR.Window}(
    (Input(:in_val => i) => inwindow for i=1:2)...,
    (Input(:obs => i) => obswindow for i=1:2)...
)))

circuit = implement_deep(with_implemented_subcomponents, Spiking())

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
# draw_spiketrain_figure(
#     collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
# )

### Below here is some testing that outputted spike counts look right

event_vecs = [
    SpikingSimulator.simulate_for_time_and_get_events(circuit, 50;
        initial_inputs=(:in_val => 1, :obs => 1)
    )
    for _=1:1000
]

dicts = [
    spiketrain_dict(
        filter(ev) do (t, compname, event)
            (compname === :ss || compname === nothing) && event isa Sim.OutputSpike
        end
    ) for ev in event_vecs
]

lens = [
    haskey(dict, :prob) ? length(dict[:prob]) : nothing for dict in dicts
]

function score_manual_count(dict::Dict, sample)
    i1 = 1
    i2 = 1
    total = 0
    cnt = 0
    while total < K
        i = length(dict[1]) ≥ i1 ? (length(dict[2]) ≥ i2 ? argmin([dict[1][i1], dict[2][i2]]) : 1) : 2
        if i == 1
            i1 += 1
        else
            i2 += 1
        end
        if sample == i
            cnt += 1
        end
        total += 1
    end
    return cnt
end
score_manually_count(dicts::Vector{<:Dict}, sample) = [score_manual_count(dict, sample) for dict in dicts]
mcnts = score_manually_count(dicts, 1)
sum(lens)