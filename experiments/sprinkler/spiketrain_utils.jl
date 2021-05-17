using SpikingCircuits.SpiketrainViz

# Spiketrain of MH Outputs

function mh_output_spiketrain_dict(output_event_vector)
    spiketrains = Dict()
    for (time, compname, outspike) in output_event_vector
        @assert compname === nothing
        @assert outspike.name isa Pair && outspike.name.first == :updated_traces
        @assert outspike.name.second isa Pair && outspike.name.second.first isa Int
        sample_idx = outspike.name.second.first
        valname_and_val = outspike.name.second.second

        @assert valname_and_val isa Pair
        valname = nothing
        while valname_and_val isa Pair
            valname = Circuits.nest(valname, valname_and_val.first)
            valname_and_val = valname_and_val.second
        end
        val = valname_and_val

        key = (sample_idx, valname, val)
        if haskey(spiketrains, key)
            push!(spiketrains[key], time)
        else
            spiketrains[key] = [time]
        end
    end
    return spiketrains
end

function get_names_and_trains(dict)
    mh_sample_indices = Set(key[1] for key in keys(dict))
    @assert all((i in mh_sample_indices) for i=1:length(mh_sample_indices))
    mh_sample_indices = 1:length(mh_sample_indices)
    valnames = Set(key[2] for key in keys(dict))
    vals = Dict(
        valname => Set(
            key[3] for key in keys(dict)
            if  key[2] == valname
        )
        for valname in valnames
    )
    names = []
    spiketrains = []
    display(dict)
    for idx=1:length(mh_sample_indices)
        @assert idx in mh_sample_indices
        for valname in valnames
            for val in vals[valname]
                if haskey(dict, (idx, valname, val))
                    push!(names, "$idx | $valname = $val")
                    push!(spiketrains, dict[(idx, valname, val)])
                else
                    println("Could not find key $((idx, valname, val))")
                end
            end
        end
    end
    return (names, spiketrains)
end

function draw_mh_figure(dict::Dict; endtime=nothing)
    names, trains = get_names_and_trains(dict)
    if !isnothing(endtime)
        draw_spiketrain_figure(trains; names=names, xmin=0, xmax=endtime)
    else
        draw_spiketrain_figure(trains; names=names, xmin=0)
    end
end
draw_mh_figure(events::Vector; endtime=nothing) = draw_mh_figure(
    mh_output_spiketrain_dict(filter(is_root_outspike, events)); endtime
)

is_root_outspike((t, compname, evt)) = 
    compname === nothing && evt isa SpikingSimulator.OutputSpike

# Spiketrain of some random neurons
function get_neuron_compnames(events)
    compnames = Set()
    prefixes = Set()
    for (_, compname, _) in events
        push!(compnames, compname)
        if compname isa Pair
            push!(prefixes, compname.first)
        end
    end

    neuron_compnames = filter(compnames) do name; !(name in prefixes); end
    return neuron_compnames
end

function get_dict(events, compnames_to_show; tmin=-Inf, tmax=Inf)
    name_to_time = Dict()
    for (t, c, _) in events
        if c in compnames_to_show && tmin ≤ t ≤ tmax
            push!(get!(name_to_time, c, Float64[]), t)
        end
    end
    return name_to_time
end

using Random: shuffle!

function draw_random_neuron_figure(events::Vector; endtime=nothing, kwargs...)
    (names, trains) = get_random_neuron_names_and_trains(events; kwargs...)
    draw_random_neuron_figure(trains, names; endtime)
end
function draw_random_neuron_figure(trains::Vector, names::Vector; endtime=nothing)
    if !isnothing(endtime)
        draw_spiketrain_figure(trains; names=names, xmin=0, xmax=endtime)
    else
        draw_spiketrain_figure(trains; names=names, xmin=0)
    end
end
function get_random_neuron_names_and_trains(events; num_neurons=50, tmin=-Inf, tmax=Inf)
    neuron_compnames = collect(get_neuron_compnames(events))
    shuffle!(neuron_compnames)
    selected = Set(neuron_compnames[1:num_neurons])
    dict = get_dict(events, selected; tmin, tmax)

    names = [" " for name in keys(dict)]
    trains = [train::Vector{Float64} for train in values(dict)]
    return (names, trains)
end

### Extract MH states
function get_mh_states(events)
    dict = mh_output_spiketrain_dict(filter(is_root_outspike, unblocked_events))
    sequence = []
    for ((kernelidx, variable, value), times) in dict
        for time in times
            push!(sequence, (time, (variable, value)))
        end
    end
    sort!(sequence, by=((t, k),) -> t)
    assmts = []
    for ((_, v1), (_, v2), (_, v3)) in Iterators.partition(sequence, 3)
        @assert Set([v1[1], v2[1], v3[1]]) == Set([:sprinkler, :raining, :grasswet])
        push!(assmts, (;(v[1] => bool_val(v[2]) for v in (v1, v2, v3))...))
    end
    return assmts
end
bool_val(x) = x == 1 ? true : x == 2 ? false : error("Unexpected value: $x.")