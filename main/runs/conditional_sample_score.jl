import JSON

using Circuits
using SpikingCircuits
const Sim = SpikingSimulator

includet("../components/value_types.jl")

includet("../components/mux/mux.jl")
includet("../components/ipoisson_gated_repeater.jl")
includet("../components/mux/int_poisson_mux.jl")
includet("../components/cvb.jl")
includet("../components/conditional_sample_score/abstract.jl")
includet("../components/conditional_sample_score/spiking.jl")

unit = SpikingConditionalSampleScore(
    ConditionalSampleScore(
        # col j =  P[x | (Y=j)]
        [0.75 0.25;
         0.25  0.75],
        true
    ),
    0.01,
    2.0
)

circuit = implement_deep(unit, Spiking())

events = Sim.simulate_for_time_and_get_events(circuit, 5.0;
    initial_inputs=(:in_val => 2,)
)

includet("../visualization/circuit_visualization/animation_interface.jl")
try
    open("visualization/circuit_visualization/frontend/renders/joint_anim.json", "w") do f
        JSON.print(f, animation_to_frontend_format(Sim.initial_state(circuit), events), 2)
    end
    println("Wrote animation file.")
catch e
    @warn "Failed to write animation file: $e"
end

##############
# Spiketrain #
##############

# outputname => time
function spiketrain_dict(event_vector)
    spiketrains = Dict()
    for (time, _, outspike) in event_vector
        if haskey(spiketrains, outspike.name)
            push!(spiketrains[outspike.name], time)
        else
            spiketrains[outspike.name] = [time]
        end
    end
    return spiketrains
end

includet("../visualization/spiketrain.jl")
using .SpiketrainViz

is_primary_output(compname, event) = (isnothing(compname) && event isa Sim.OutputSpike)
dict = spiketrain_dict(filter(((t,args...),) -> is_primary_output(args...), events))
draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)




# implement_deep(
#         implement(
#             implement(unit, Spiking()), Spiking(),
#             :spikers
#         ), Spiking(),
#         :cvb
#     )

includet("../visualization/circuit_visualization/component_interface.jl")

open("visualization/circuit_visualization/frontend/renders/joint.json", "w") do f
    JSON.print(f, viz_graph(circuit), 2)
end
println("Wrote component viz file.")

# end