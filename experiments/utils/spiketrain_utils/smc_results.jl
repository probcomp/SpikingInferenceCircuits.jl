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
        valname => sort(collect(Set(
            key[3] for key in keys(dict)
            if  key[2] == valname
        )))
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

# function draw_smc_figure(dict::Dict; endtime=nothing)
#     names, trains = smc_get_names_and_trains(dict)
#     if !isnothing(endtime)
#         draw_spiketrain_figure(trains; names=names, xmin=0, xmax=endtime)
#     else
#         draw_spiketrain_figure(trains; names=names, xmin=0)
#     end
# end
# draw_smc_figure(events::Vector; endtime=nothing) = draw_smc_figure(
#     smc_output_spiketrain_dict(filter(is_root_outspike, events)); endtime
# )

is_root_outspike((t, compname, evt)) = 
    compname === nothing && evt isa SpikingSimulator.OutputSpike


# Get states
function get_smc_states(events, nparticles, nlatents)
    dict = smc_output_spiketrain_dict(filter(is_root_outspike, events))
    sequence = []
    for ((particle_idx, variable, value), times) in dict
        for time in times
            push!(sequence, (time, (particle_idx, variable, value)))
        end
    end
   
    sort!(sequence, by=((t, k),) -> t)
    idx_var_vals_full = [idx_var_val for (_, idx_var_val) in sequence]
    
    particle_approximations = []

    for idx_var_vals in Iterators.partition(idx_var_vals_full, nlatents * nparticles)
        latents = Set(varname for (_, varname, _) in idx_var_vals)
        satisfies_expected_conditions = (
            length(latents) == nlatents &&
            Set(
                (part_idx, varname) for (part_idx, varname, _) in idx_var_vals
            ) == Set(
                Iterators.product(1:nparticles, latents)
            )
        )
        if !satisfies_expected_conditions
            @warn "Latents did not satisfy expected conditions!  Returning particle approximations obtained until this erroneous timestep."
            return particle_approximations
        end

        assmts_t = Dict(
            varname => [-1 for _=1:nparticles]
            for varname in latents
        ) # assmts[varname] = [val_in_particle1, ..., val_in_particleN]
        for (idx, varname, varval) in idx_var_vals
            assmts_t[varname][idx] = varval
        end

        push!(particle_approximations, assmts_t)
    end
    
    return particle_approximations
end
