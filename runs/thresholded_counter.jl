import JSON

using Circuits
using SpikingCircuits
const Sim = SpikingSimulator
using Distributions: Categorical

includet("../src/value_types.jl")
includet("../src/components/mux/mux.jl")
includet("../src/components/ipoisson_gated_repeater.jl")
includet("../src/components/mux/int_poisson_mux.jl")
includet("../src/components/cvb.jl")
includet("../src/components/conditional_sample_score/abstract.jl")
includet("../src/components/conditional_sample_score/spiking.jl")
includet("../src/components/thresholded_spike_counter.jl")
includet("../src/components/to_assmts/abstract.jl")
includet("../src/components/to_assmts/spiking.jl")
includet("../src/cpt.jl")
includet("../src/components/cpt_sample_score/abstract.jl")
includet("../src/components/cpt_sample_score/spiking.jl")

unit = ThresholdedCounter(1)
circuit = implement_deep(unit, Spiking())
println("Circuit constructed.")
events = Sim.simulate_for_time_and_get_events(circuit,  1.0; initial_inputs=(:in,))
println("Events obtained.")