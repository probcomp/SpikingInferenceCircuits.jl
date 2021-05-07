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

bival = IndexedValues((SpikeWire(), SpikeWire()))
# input --> x --> y
bn(pcs) = CompositeComponent(
    NamedValues(:input => bival),
    NamedValues(
        :trace => NamedValues(:x => bival, :y => bival),
        :inverse_probs => NamedValues(:x => outputs(pcs)[:inverse_prob], :y => outputs(pcs)[:inverse_prob])
    ),
    (x=pcs, y=pcs),
    (
        Input(:input) => CompIn(:x, :in_val),
        CompOut(:x, :value) => Output(:trace => :x),
        CompOut(:y, :value) => Output(:trace => :y),
        CompOut(:x, :value) => CompIn(:y, :in_val),
        CompOut(:x, :inverse_prob) => Output(:inverse_probs => :x),
        CompOut(:y, :inverse_prob) => Output(:inverse_probs => :y)
    )
)

inwindow = PulseIR.Window(
    PulseIR.Interval(0., 0.),
    Inf, Inf
)
d1 = PulseIR.output_windows(bn(pcs), Dict{Input, PulseIR.Window}(
    Input(:input => i) => inwindow for i=1:2
))
display(d1)

with_implemented_subcomponents = SDCs.PulseConditionalSample(
    PoissonStreamSamples(pcs.streamsamples, 1/(10^10)),
    PoissonAsyncOnGate(pcs.mux_on_gate, 12),
    PoissonOffGate(pcs.wta_offgate, 12),
    PoissonThresholdedIndicator(pcs.ti, 30),
    PoissonOffGate(pcs.offgate, 12),
    50.0, .1
)

d2 = PulseIR.output_windows(bn(with_implemented_subcomponents), Dict{Input, PulseIR.Window}(
    Input(:input => i) => inwindow for i=1:2
))
display(d2)

@assert d1 == d2

### test in simulation:

impl = implement_deep(bn(with_implemented_subcomponents), Spiking())

events = SpikingSimulator.simulate_for_time_and_get_events(impl, 500;
    initial_inputs=(:input => 1,)
)
|
# TODO: stop copying this definition everywhere!!!
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
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        compname === nothing && event isa Sim.OutputSpike
    end
)

using SpikingCircuits.SpiketrainViz
draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)
