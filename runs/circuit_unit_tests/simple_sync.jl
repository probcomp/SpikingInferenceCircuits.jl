using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
const Sim = SpikingCircuits.SpikingSimulator

Circuits.implement(s::PulseIR.Sync, ::Spiking) =
    PulseIR.PoissonSync(
        s.cluster_sizes,
        (1000, 20),
        (0.1, 1000, 20),
        (50., 4, (0.1, 1000, 20), 0., 100.)
    )

sync = PulseIR.Sync([2, 1, 1])
impl = implement_deep(sync, Spiking())
println("Implemented.")

get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(impl, 500.;
    initial_inputs=(1 => 1, 2 => 1, 3 => 1)
)
events = get_events(impl)
println("Simulation completed")

include("spiketrain_utils.jl")
