using SpikingCircuits.SpiketrainViz

function extract_valname_and_val(valname_and_val)
    @assert valname_and_val isa Pair
    valname = nothing
    while valname_and_val isa Pair
        valname = Circuits.nest(valname, valname_and_val.first)
        valname_and_val = valname_and_val.second
    end
    val = valname_and_val
    return (valname, val)
end

function smc_output_spiketrain_dict(output_event_vector)
    spiketrains = Dict()
    for (time, compname, outspike) in output_event_vector
        @assert compname === nothing
        @assert outspike.name isa Pair && outspike.name.first isa Int
        particle_idx = outspike.name.first
        (valname, val) = extract_valname_and_val(outspike.name.second)

        key = (particle_idx, valname, val)
        if haskey(spiketrains, key)
            push!(spiketrains[key], time)
        else
            spiketrains[key] = [time]
        end
    end
    return spiketrains
end

function smc_get_names_and_trains(dict)
    smc_particle_indices = Set(key[1] for key in keys(dict))
    @assert all((i in smc_particle_indices) for i=1:length(smc_particle_indices))
    smc_particle_indices = 1:length(smc_particle_indices)
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
    for idx=1:length(smc_particle_indices)
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

function draw_smc_figure(dict::Dict; endtime=nothing)
    names, trains = smc_get_names_and_trains(dict)
    if !isnothing(endtime)
        draw_spiketrain_figure(trains; names=names, xmin=0, xmax=endtime)
    else
        draw_spiketrain_figure(trains; names=names, xmin=0)
    end
end
draw_smc_figure(events::Vector; endtime=nothing) = draw_mh_figure(
    smc_output_spiketrain_dict(filter(is_root_outspike, events)); endtime
)

is_root_outspike((t, compname, evt)) = 
    compname === nothing && evt isa SpikingSimulator.OutputSpike