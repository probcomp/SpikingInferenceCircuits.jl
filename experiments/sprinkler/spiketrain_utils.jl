using SpikingCircuits.SpiketrainViz

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
    valnames = Set(key[2] for key in keys(dict) if key[1] == first(mh_sample_indices))
    vals = Dict(
        valname => Set(key[3] for key in keys(dict)
        if key[1] == first(mh_sample_indices) && key[2] == valname)
        for valname in valnames
    )
    names = []
    spiketrains = []
    for idx=1:length(mh_sample_indices)
        @assert idx in mh_sample_indices
        for valname in valnames
            for val in vals[valname]
                push!(names, "$idx | $valname = $val")
                push!(spiketrains, dict[(idx, valname, val)])
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