module Test
import JSON

include("main.jl")
import .Circuits: implement_deep, Spiking, CompositeValue, CompositeComponent, IndexedComponentGroup, PoissonNeuron, Input, CompIn, CompOut, Output, SpikeWire
const Sim = Circuits.SpikingSimulator

comp() = CompositeComponent(
    CompositeValue((in=SpikeWire(),)), CompositeValue((out=SpikeWire(),)), (
        IndexedComponentGroup(PoissonNeuron(1.0) for _=1:4),
    ), (CompOut(1 => i, :out) => CompIn(1 => j, :off) for i=1:4, j=1:4)
) |> (c -> implement_deep(c, Spiking()))

component = comp()
println("Made component!")

open("visualization/frontend/renders/simple_test.json", "w") do f
    JSON.print(f, Circuits.viz_graph(component), 2)
end
println("Wrote component viz file.")

initial_state() = Sim.CompositeState(
    (Sim.CompositeState(
        Tuple(Sim.OnOffState(true) for _=1:4)
    ),)
)

sim(component) = Sim.simulate_for_time_and_get_spikes_and_primitive_statechanges(component, initial_state(), 1.0)
events = sim(component)
println("Simulation 1 completed!")

# @time sim(component)
# Sim.simulate_for_time_and_get_spikes_and_primitive_statechanges(component, initial_state, 1.0)
# println("Simulation 2 completed!")

open("visualization/frontend/renders/simple_test_animation.json", "w") do f
    JSON.print(f, Circuits.animation_to_frontend_format(initial_state, events), 2)
end
println("Wrote animation file.")
end # module