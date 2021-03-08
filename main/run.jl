module Test
import JSON

include("main.jl")
const Sim = SNNs.SpikingSimulator

spiking_sampler = SNNs.PoissonRaceCatSampler(SNNs.Categorical([0.1, 0.2, 0.2, 0.5]), 1.0)
graph = SNNs.implement(spiking_sampler, SNNs.Spiking())

initial_state = Sim.CompositeState(Tuple(Sim.OnOffState(true) for _=1:4))
frames = Sim.simulate_for_time_and_get_spikes(graph, initial_state, 5.0)
open("visualization/frontend/animation.json", "w") do f
    JSON.print(f, SNNs.animation_to_frontend_format(frames), 2)
end


# open("visualization/frontend/testgraph2.json", "w") do f
#     JSON.print(f, SNNs.viz_graph(graph), 2)
# end

end