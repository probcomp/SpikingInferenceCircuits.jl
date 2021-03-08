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
# display(comp)
println("made component")

open("visualization/frontend/conc_samp.json", "w") do f
    JSON.print(f, Circuits.viz_graph(comp), 2)
end

end