import JSON

using Circuits
using SpikingCircuits
const Sim = SpikingSimulator
using Distributions: Categorical

includet("../components/value_types.jl")
includet("../components/mux/mux.jl")
includet("../components/ipoisson_gated_repeater.jl")
includet("../components/mux/int_poisson_mux.jl")
includet("../components/cvb.jl")
includet("../components/conditional_sample_score/abstract.jl")
includet("../components/conditional_sample_score/spiking.jl")
includet("../components/thresholded_spike_counter.jl")
includet("../components/to_assmts/abstract.jl")
includet("../components/to_assmts/spiking.jl")
includet("../components/cpt.jl")
includet("../components/cpt_sample_score/abstract.jl")
includet("../components/cpt_sample_score/spiking.jl")

cpt = CPT(
    map(Categorical, [
        [[0.5, 0.5]] [[0.25, 0.75]];
        [[0.98, 0.02]] [[0.1, 0.9]]
    ])
)

unit = SpikingCPTSampleScore(
    CPTSampleScore(cpt, true),
    0.0001,
    2.0
)

circuit = implement_deep(unit, Spiking())
println("Circuit constructed.")

includet("../visualization/circuit_visualization/component_interface.jl")

open("visualization/circuit_visualization/frontend/renders/cpt.json", "w") do f
    JSON.print(f, viz_graph(circuit), 2)
end
println("Wrote component viz file.")

events = Sim.simulate_for_time_and_get_events(circuit,  16.0;
    initial_inputs=(:in_vals => 1 => 2, :in_vals => 2 => 1)
)
println("Simulation run.")

includet("../visualization/circuit_visualization/animation_interface.jl")
try
    open("visualization/circuit_visualization/frontend/renders/cpt_anim.json", "w") do f
        JSON.print(f, animation_to_frontend_format(Sim.initial_state(circuit), events), 2)
    end
    println("Wrote animation file.")
catch e
    @warn "Failed to write animation file: $e"
end

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

includet("../visualization/spiketrain.jl")
using .SpiketrainViz

is_primary_output(compname, event) = (isnothing(compname) && event isa Sim.OutputSpike)
dict = spiketrain_dict(filter(((t,args...),) -> is_primary_output(args...), events))
draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)

###
# Multi-run analysis 
###

RUN_LEN() = 16.0

function do_run(assmt...; obs=nothing)
    events_vecs = [
        Sim.simulate_for_time_and_get_events(circuit, RUN_LEN();
            initial_inputs=(:in_vals => 1 => assmt[1], :in_vals => 2 => assmt[2],
                (obs === nothing ? () : (:obs => obs,))...
            )
        )
        for _=1:50
    ]

    return events_vecs
end

get_rate(dict) =
    if haskey(dict, :prob)
        times = dict[:prob]
        times_in_second_half = filter(x -> x > RUN_LEN() / 2, times)
        length(times_in_second_half) / (RUN_LEN() / 2)
    else
        0
    end

function count(samples)
    cnts = Dict()
    for smp in samples
        cnts[smp] = get(cnts, smp, 0) + 1
    end
    return cnts |> (d -> [d[i] for i=1:length(d)])
end
function avg_rates(samples, rates)
    rts = Dict()
    for (smp, rt) in zip(samples, rates)
        rts[smp] = push!(get(rts, smp, Float64[]), rt)
    end
    Dict(n => sum(rt)/length(rt) for (n, rt) in pairs(rts))
end

assmt = (2, 1)
println("Expected probs: $(cpt[assmt...] |> probs)")
events_vecs = do_run(assmt...)

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

println("sample counts: $(count(samples))")

rates = map(get_rate, spiketrain_dicts)

avg_probs = Dict(
    n => (r / outputs(unit)[:prob].reference_rate)
    for (n, r) in avg_rates(samples, rates)
) |> (d -> [d[i] for i=1:length(d)])
println("average probs: $avg_probs")