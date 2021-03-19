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
        # row j =  P[y | (X=j)]
        # [0.75 0.25;
        #  0.25  0.75],
        #  [0.2 0.8; 0.2 0.8],
        [0.6 0.25 0.15;
        0.05 0.75 0.20;
        0.90 0.00 0.10;
        0.90 0.09 0.01],
        true
    ),
    0.01,
    2.0
)

circuit = implement_deep(unit, Spiking())

# events = Sim.simulate_for_time_and_get_events(circuit, 5.0;
#     initial_inputs=(:in_val => 2,)
# )

# includet("../visualization/circuit_visualization/animation_interface.jl")
# try
#     open("visualization/circuit_visualization/frontend/renders/joint_anim.json", "w") do f
#         JSON.print(f, animation_to_frontend_format(Sim.initial_state(circuit), events), 2)
#     end
#     println("Wrote animation file.")
# catch e
#     @warn "Failed to write animation file: $e"
# end

# ##############
# # Spiketrain #
# ##############

# # outputname => time
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

# includet("../visualization/spiketrain.jl")
# using .SpiketrainViz

is_primary_output(compname, event) = (isnothing(compname) && event isa Sim.OutputSpike)
# dict = spiketrain_dict(filter(((t,args...),) -> is_primary_output(args...), events))
# draw_spiketrain_figure(
#     collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
# )

# includet("../visualization/circuit_visualization/component_interface.jl")

# open("visualization/circuit_visualization/frontend/renders/joint.json", "w") do f
#     JSON.print(f, viz_graph(circuit), 2)
# end
# println("Wrote component viz file.")

#####
# Analysis to check sampling probabilities & outputted probabilities look right
#####

RUN_LEN() = 6.0
events_vecs = [
    Sim.simulate_for_time_and_get_events(circuit, RUN_LEN();
        initial_inputs=(:in_val => 4,)
    )
    for _=1:100
]
spiketrain_dicts = [
    spiketrain_dict(
        filter(((t,args...),) -> is_primary_output(args...), event_vec)
    ) for event_vec in events_vecs
]

samples = map(spiketrain_dicts) do dict
    [if key isa Pair && key.first == :sample
        key.second
    elseif key != :prob
        error("unexpected: $key")
    end
    for key in keys(dict)][1]
end

rates = map(spiketrain_dicts) do dict
    if haskey(dict, :prob)
        length(filter(x->x > RUN_LEN()/2, dict[:prob])) / (RUN_LEN()/2)
    else
        0
    end
end

function count(samples)
    cnts = Dict()
    for smp in samples
        cnts[smp] = get(cnts, smp, 0) + 1
    end
    return cnts
end
function avg_rates(samples, rates)
    rts = Dict()
    for (smp, rt) in zip(samples, rates)
        rts[smp] = push!(get(rts, smp, Float64[]), rt)
    end
    Dict(n => sum(rt)/length(rt) for (n, rt) in pairs(rts))
end

# println("samples: $samples")
# println("rates: $(rates ./ outputs(unit)[:prob].reference_rate)")
println("sample counts: $(count(samples))")
# println("average rates: $(avg_rates(samples, rates))")
avg_probs = Dict(
    n => r / outputs(unit)[:prob].reference_rate
    for (n, r) in avg_rates(samples, rates)
)
println("average probs: $avg_probs")


# end