include("main.jl")

spiking_sampler = PoissonRaceCatSampler(Categorical([0.1, 0.2, 0.2, 0.5]), 1.0)
graph = implement(spiking_sampler, Spiking())

open("visualization/frontend/testgraph.json", "w") do f
    JSON.print(f, JSON.parse(json_graph(graph)), 2)
end