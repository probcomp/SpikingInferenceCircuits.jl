module Test
import JSON

include("main.jl")
const Sim = SNNs.SpikingSimulator

abstract_sampler = SNNs.AbstractCatSamplerWithProb(SNNs.Categorical([0.1, 0.2, 0.2, 0.5]))

# initial_state = Sim.CompositeState(Tuple(Sim.OnOffState(true) for _=1:4))
# frames = Sim.simulate_for_time_and_get_spikes(graph, initial_state, 5.0)
# open("visualization/frontend/animation.json", "w") do f
#     JSON.print(f, SNNs.animation_to_frontend_format(frames), 2)
# end


open("visualization/frontend/abs_samp.json", "w") do f
    JSON.print(f, SNNs.viz_graph(abstract_sampler), 2)
end

end