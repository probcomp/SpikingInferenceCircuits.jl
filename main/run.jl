include("main.jl")

spiking_sampler = SNNs.PoissonRaceCatSampler(SNNs.Categorical([0.1, 0.2, 0.2, 0.5]), 1.0)
graph = SNNs.implement(spiking_sampler, SNNs.Spiking())

open("visualization/frontend/testgraph2.json", "w") do f
    JSON.print(f, SNNs.viz_graph(graph), 2)
end