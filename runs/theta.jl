using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using .SDCs: IndicatedSpikeCountReal, UnbiasedSpikeCountReal

theta = SDCs.PulseTheta(
    #                             M     L    ΔT    rate
    10, 2, PulseIR.PoissonTheta, 1000, -20., 400., 1.0,
    PulseIR.PoissonOffGate(
        PulseIR.ConcreteOffGate(400. #= ΔT =#, 0.2, 500), 20
    ),
    PulseIR.PoissonThresholdedIndicator,
    (400., 0.2, 500, 20)
)

println("Theta constructed.")

implemented = implement_deep(theta, Spiking())

println("Implemented deeply.")

get_events(impl; log=true) = SpikingSimulator.simulate_for_time_and_get_events(
    impl, 50.0; initial_inputs=(
        (1 => :count for _=1:4)..., 1 => :ind, 2 => :count, 2 => :ind
    ), log
)

events = get_events(implemented, log=false)
println("Simulation completed.")

# include("spiketrain_utils.jl")