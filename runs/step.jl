using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
const Sim = SpikingCircuits.SpikingSimulator

### Implementation Rules ###

Circuits.implement(ta::SIC.SDCs.ToAssmts, ::Spiking) =
    SDCs.PulseToAssmts(
        ta, PulseIR.PoissonThresholdedIndicator,
        # ΔT, max_delay, M, R
        (500, 0.5, 50, 40)
        # Note that the R needs to be high since getting spikes while off is catastrophic.
        # TODO: Design things so this is not catastrophic (or can't happen at
        # realistic rates)!
    )

K = 10
ONRATE = 0.2
Circuits.implement(cs::SIC.SDCs.ConditionalSample, ::Spiking) =
    SDCs.PoissonPulseConditionalSample(
        (cs, K, ONRATE,
            500, # ΔT
            0.2, # max_delay
            1000, # M (num spikes to override offs/ons)
            50, # max delay before sample is emitted
            0.1 # intersample hold
        ),
        10^(-10), 12
    )
Circuits.implement(cs::SIC.SDCs.ConditionalScore, ::Spiking) =
    SDCs.PoissonPulseConditionalScore((cs, K, ONRATE, 500, 0.2, 1000), 10^(-10), 12)

Circuits.implement(lt::SIC.SDCs.LookupTable, ::Spiking) =
    SIC.SDCs.OneHotLookupTable(lt)

Circuits.implement(::Binary, ::Spiking) = SpikeWire()

to_spiking_real(::SDCs.SingleNonnegativeReal) = 
    SDCs.IndicatedSpikeCountReal(SDCs.UnbiasedSpikeCountReal(K))
to_spiking_real(v::SDCs.ProductNonnegativeReal) =
    SDCs.ProductNonnegativeReal(map(to_spiking_real, v.factors))
to_spiking_real(v::SDCs.NonnegativeReal) = to_spiking_real(implement(v, Spiking()))
Circuits.implement(v::SDCs.SingleNonnegativeReal, ::Spiking) = to_spiking_real(v)

Circuits.implement(s::PulseIR.Sync, ::Spiking) =
    PulseIR.PoissonSync(
        s.cluster_sizes,
        (1000., 30.),
        (0.1, 30.),
        (50., 4, (0.1, 1000, 20), 0., 100.)
    )

### Gen Fn Compilation ###

@gen (static) function test(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input)
    y ~ CPT([[0.9, 0.1], [0.1, 0.9]])(x)
    z ~ CPT([
        [[0.9, 0.1]] [[0.1, 0.9]];
        [[0.75, 0.25]] [[0.25, 0.75]]
    ])(x, y)
    return z
end

propose_circuit = gen_fn_circuit(test, (input=FiniteDomain(2),), Propose())
implemented_propose = implement_deep(propose_circuit, Spiking())

println("Got implemented propose.")

comp = CompositeComponent(
    NamedValues(:in => inputs(propose_circuit)),
    NamedValues(:out => outputs(propose_circuit)),
    (
        prop=propose_circuit,
        step=SDCs.Step(outputs(propose_circuit))
    ),
    (
        Input(:in => :inputs) => CompIn(:prop, :inputs),
        CompOut(:prop, :trace) => CompIn(:step, :in => :trace),
        CompOut(:prop, :score) => CompIn(:step, :in => :score),
        CompOut(:prop, :value) => CompIn(:step, :in => :value),
        CompOut(:step, :out) => Output(:out)
    )
)
println("comp constructed")
implemented = implement_deep(comp, Spiking())
println("Implemented composite component deep.")

get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(impl, 1000.;
    initial_inputs=(:in => :inputs => :input => 1,)
)
events = get_events(implemented)
println("Simulation run.")

include("spiketrain_utils.jl")

draw_fig(events)