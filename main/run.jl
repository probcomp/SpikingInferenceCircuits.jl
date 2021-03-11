module Test
import JSON

include("main.jl")
using .Circuits: AbstractCatSamplerWithProb, implement_deep, Spiking, PoissonRace_MultiProbNeuron_CatSampler
const Sim = Circuits.SpikingSimulator




# non-standard implementations we will use:
Circuits.implement(s::AbstractCatSamplerWithProb, ::Spiking) =
    PoissonRace_MultiProbNeuron_CatSampler(s, 1.0, 4)
Circuits.implement(v::Circuits.FiniteDomainValue, ::Spiking) =
    Circuits.SpikingCategoricalValue(v.n)

abstract_sampler = AbstractCatSamplerWithProb(Circuits.Categorical([0.1, 0.2, 0.2, 0.5]))

# initial_state = Sim.CompositeState(Tuple(Sim.OnOffState(true) for _=1:4))
# frames = Sim.simulate_for_time_and_get_spikes(graph, initial_state, 5.0)
# open("visualization/frontend/animation.json", "w") do f
#     JSON.print(f, Circuits.animation_to_frontend_format(frames), 2)
# end
comp = concrete_sampler = Circuits.implement_deep(abstract_sampler, Spiking())
println("Made component!")

open("visualization/frontend/renders/conc_samp.json", "w") do f
    JSON.print(f, Circuits.viz_graph(comp), 2)
end
println("Wrote component viz file.")

events = Sim.simulate_for_time_and_get_spikes_and_primitive_statechanges(comp, 5.0; initial_inputs=(:on,))
println("Simulation completed!")

open("visualization/frontend/renders/conc_samp_animation.json", "w") do f
    JSON.print(f, Circuits.animation_to_frontend_format(Sim.initial_state(comp), events), 2)
end
println("Wrote animation file.")

#==
TODO: the errors now are probably being caused because I don't think the code can support a component output
going into a component input (nor a compout going into a component input).

I should think through which of these we need to support, and fix the parts of the code which disallow it.
The simulator is part of it; I don't remember if other parts of the code will need fixing as well.
=#


# comp = Circuits.implement_deep(
#     Circuits.IndexedComponentGroup(Circuits.PoissonNeuron(1.0) for _=1:2),
#     Spiking()
# )

# comp = Circuits.implement(Circuits.PoissonRaceCatSampler(Circuits.Categorical([0.1,0.2,0.2,0.5]), 1.0), Spiking())

end