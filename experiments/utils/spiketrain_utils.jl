function get_name_hierarchy(names)
    d = Dict()
    for name in names
        if name isa Pair
            d[name.first] = push!(get(d, name.first, Set()), name.second)
        else
            get!(d, name, Set())
        end
    end
    return Dict( k => (isnothing(v) ? nothing : get_name_hierarchy(v)) for (k, v) in d)
end

include("spiketrain_utils/get_name_hierarchy.jl")

# include("spiketrain_utils/smc_results.jl")
include("spiketrain_utils/internal_spiketrains.jl")