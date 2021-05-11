using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using .SDCs: IndicatedSpikeCountReal, UnbiasedSpikeCountReal

theta = SDCs.PulseTheta(
    10,2, PulseIR.PoissonTheta, 1000, 400., 100.,
    PulseIR.PoissonOffGate(
        PulseIR.ConcreteOffGate(400. #= ΔT =#, 0.2, 500), 20
    ),
    PulseIR.PoissonThresholdedIndicator,
    (400., 0.2, 500, 20)
)

println("Theta constructed.")

implemented = implement_deep(theta, Spiking())

println("Implemented deeply.")

get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(
    impl, 500.0; initial_inputs=(
        (1 => :count for _=1:4)..., 1 => :ind, 2 => :count, 2 => :ind
    )
)

events = get_events(implemented)
println("Simulation completed.")

include("spiketrain_utils.jl")