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

### Sync ###
cluster_addrs = [
    :trace => :x, :trace => :y, :trace => :z, (:score => key for key in keys_deep(outputs(implemented_propose)[:score]))...
]
cluster_sizes = [
    let val = outputs(implemented_propose)[addr]
        val isa CompositeValue ? length(collect(keys_deep(val))) : 1
    end
    for addr in cluster_addrs
]
sync = PulseIR.Sync(cluster_sizes)


### Implementation rules ###
Circuits.implement(s::PulseIR.Sync, ::Spiking) =
    PulseIR.PoissonSync(
        s.cluster_sizes,
        (1000., 30.),
        (0.1, 30.),
        (50., 4, (0.1, 1000, 20), 0., 100.)
    )

# implement
implemented_sync = implement_deep(sync, Spiking())
println("Got implemented sync.")

### Synchronize the output of the propose circuit!
composite = CompositeComponent(
    inputs(implemented_propose), outputs(implemented_propose),
    (
        propose=implemented_propose,
        sync=sync
    ),
    (
        (
            Input(addr) => CompIn(:propose, addr)
            for addr in keys_deep(inputs(implemented_propose))
        )...,
        (
            CompOut(:propose, addr) => CompIn(:sync, i)
            for (i, addr) in enumerate(cluster_addrs)
            if outputs(implemented_propose)[addr] isa CompositeValue
        )...,
        (
            CompOut(:propose, addr) => CompIn(:sync, i => 1)
            for (i, addr) in enumerate(cluster_addrs)
            if outputs(implemented_propose)[addr] isa SpikeWire
        )...,
        (
            CompOut(:sync, i) => Output(addr)
            for (i, addr) in enumerate(cluster_addrs)
            if outputs(implemented_propose)[addr] isa CompositeValue
        )...,
        (
            CompOut(:sync, i => 1) => Output(addr)
            for (i, addr) in enumerate(cluster_addrs)
            if outputs(implemented_propose)[addr] isa SpikeWire
        )...
    )
)
println("composite constructed.")

implemented = implement_deep(composite, Spiking())

println("composite implemented.")

get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(impl, 1000.;
    initial_inputs=(:inputs => :input => 1,)
)
events = get_events(implemented)
println("Simulation run.")

include("spiketrain_utils.jl")

draw_fig(events)