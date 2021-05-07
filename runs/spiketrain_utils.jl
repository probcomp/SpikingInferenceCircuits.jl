using SpikingCircuits.SpiketrainViz

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

out_st_dict(events) = filter(events) do (t, compname, event)
    (compname === :ss || compname === nothing) && event isa SpikingSimulator.OutputSpike
end |> spiketrain_dict

function draw_fig(events)
    dict = out_st_dict(events)
    draw_spiketrain_figure(
        collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
    )
end