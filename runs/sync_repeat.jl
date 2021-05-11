using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
const Sim = SpikingCircuits.SpikingSimulator

Circuits.implement(s::PulseIR.Sync, ::Spiking) =
    PulseIR.PoissonSync(
        s.cluster_sizes,
        (1000., 30.),
        (0.1, 30.),
        (50., 4, (0.1, 1000, 20), 0., 100.)
    )

sync = PulseIR.Sync([1, 2])
timer = PulseIR.PoissonTimer(200., 50, (0.1, 1000, 20), 0., 300.)

# idea is: pass 
comp = CompositeComponent(
    NamedValues(:go => SpikeWire()), NamedValues(:sync_out1 => SpikeWire(), :sync_out2 => SpikeWire(), :unsync_out1 => SpikeWire(), :unsync_out2 => SpikeWire()),
    (
        sync = sync,
        t11 = timer,
        t12 = timer,
        t13 = timer,
        t21 = timer,
        t22 = timer,
        t23 = timer
    ),
    (
        Input(:go) => CompIn(:t11, :start),
        CompOut(:t11, :out) => CompIn(:t12, :start),
        CompOut(:t12, :out) => CompIn(:t13, :start),
        CompOut(:t13, :out) => Output(:unsync_out1),

        Input(:go) => CompIn(:t21, :start),
        CompOut(:t21, :out) => CompIn(:t22, :start),
        CompOut(:t22, :out) => CompIn(:t23, :start),
        CompOut(:t23, :out) => Output(:unsync_out2),
        
        
        CompOut(:t13, :out) => CompIn(:sync, 1 => 1),
        CompOut(:t23, :out) => CompIn(:sync, 2 => 1),

        CompOut(:sync, 1 => 1) => CompIn(:t11, :start),
        CompOut(:sync, 2 => 1) => CompIn(:t21, :start),
        CompOut(:sync, 1 => 1) => Output(:sync_out1),
        CompOut(:sync, 2 => 1) => Output(:sync_out2)
    )
)

impl = implement_deep(comp, Spiking())
println("Implemented.")
get_events(impl) = Sim.simulate_for_time_and_get_events(impl, 5000.; initial_inputs=(:go,))

events = get_events(impl)
println("Simulation run.")

include("spiketrain_utils.jl")