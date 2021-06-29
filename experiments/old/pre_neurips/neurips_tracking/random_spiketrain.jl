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
function get_random_neuron_names_and_trains(events; num_neurons=50, min_n_spikes=0, tmin=-Inf, tmax=Inf)
    neuron_compnames = collect(get_neuron_compnames(events))
    shuffle!(neuron_compnames)
    i = 1
    names = String[]
    trains = Vector{Float64}[]
    while length(names) < num_neurons
        selected = Set(neuron_compnames[i:i + num_neurons])
        i += num_neurons
        dict = get_dict(events, selected; tmin, tmax)
        _names = [" " for name in keys(dict)]
        _trains = [train::Vector{Float64} for train in values(dict)]
        indices = [idx for idx=1:num_neurons if length(_trains[idx]) ≥ min_n_spikes]

        append!(names, _names[indices])
        append!(trains, _trains[indices])
    end

    return (names, trains)
end

