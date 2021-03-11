module Test
import JSON

include("circuits.jl")

using .Circuits: AbstractCatSamplerWithProb, implement_deep, Spiking, PoissonRace_MultiProbNeuron_CatSampler
const Sim = Circuits.SpikingSimulator

#################
# Get Component #
#################

# non-standard implementations we will use:
Circuits.implement(s::AbstractCatSamplerWithProb, ::Spiking) =
    PoissonRace_MultiProbNeuron_CatSampler(s, 1.0, 4)
Circuits.implement(v::Circuits.FiniteDomainValue, ::Spiking) =
    Circuits.SpikingCategoricalValue(v.n)

abstract_sampler = AbstractCatSamplerWithProb(Circuits.Categorical([0.1, 0.2, 0.2, 0.5]))

comp = concrete_sampler = Circuits.implement_deep(abstract_sampler, Spiking())
println("Made component!")

# open("visualization/frontend/renders/conc_samp.json", "w") do f
#     JSON.print(f, Circuits.viz_graph(comp), 2)
# end
# println("Wrote component viz file.")

##################
# Run simulation #
##################

is_primitive_outspike(compname, event) = (
    event isa Sim.OutputSpike && comp[compname] isa Circuits.PrimitiveComponent
)
is_spike_or_primitive_statechange(compname, event) = (
    event isa Sim.Spike || comp[compname] isa Circuits.PrimitiveComponent
)
events = Sim.simulate_for_time_and_get_events(comp, 5.0; initial_inputs=(:on,), event_filter=is_spike_or_primitive_statechange)
println("Simulation completed!")

try
    open("visualization/circuit_visualization/frontend/renders/conc_samp_animation.json", "w") do f
        JSON.print(f, Circuits.animation_to_frontend_format(Sim.initial_state(comp), events), 2)
    end
    println("Wrote animation file.")
catch e
    @warn "Failed to write animation file: $e"
end

######################
# Display spiketrain #
######################

filtered = filter(((t, c, e),) -> is_primitive_outspike(c, e), events)

function spiketrain_dict(event_vector)
    spiketrains = Dict()
    for (time, neuronname, _) in event_vector
        if haskey(spiketrains, neuronname)
            push!(spiketrains[neuronname], time)
        else
            spiketrains[neuronname] = [time]
        end
    end
    return spiketrains
end
spikedict = spiketrain_dict(filtered)
println("Spiketrain dictionary:")

include("visualization/spiketrain.jl")
using .SpiketrainViz: draw_spiketrain_figure
using ColorSchemes

function collect_spiketrains(spiketrain_dict)
    order = Iterators.flatten((
        [:first_samplers => j for j=1:4],
        (Iterators.flatten((
            (:prob_timers => k,),
            (:prob_samplers => k => i for i=1:4)
        )) for k=1:4)...
    ))

    return (order, [
        get(spiketrain_dict, name, Float64[]) for name in order
    ])
end

colors() = let (c1, rest) = Iterators.peel(ColorSchemes.tableau_10[1:5])
    collect(Iterators.flatten((
        Iterators.repeated(c1, 4),
        (Iterators.repeated(c, 5) for c in rest)...
    )))
end

function drawfig()
    order, trains = collect_spiketrains(spikedict)
    @assert length(trains) == 24
    @assert length(collect(order)) == 24
    println(collect(order))
    draw_spiketrain_figure(trains; names = map(x -> "$x", order), colors=colors())
end

f = drawfig()

end