using SpikingCircuits.SpiketrainViz

function spiketrain_dict(event_vector)
    spiketrains = Dict()
    for (time, compname, outspike) in event_vector
        key = "$compname: $(outspike.name)"
        if haskey(spiketrains, key)
            push!(spiketrains[key], time)
        else
            spiketrains[key] = [time]
        end
    end
    return spiketrains
end

out_st_dict(events) = filter(events) do (t, compname, event)
    compname === nothing && event isa SpikingSimulator.OutputSpike
end |> spiketrain_dict

draw_fig(events::Vector) = draw_fig(out_st_dict(events))
draw_fig(dict::Dict) = draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)