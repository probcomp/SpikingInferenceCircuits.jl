"""
Get the hierarchy of circuit names which appear in the spikes outputted by a spiking simulation.
Takes the form of a recursively-nested Dict `h` where `h[k]` is the dict of all the name hierarchy
for the sub-circuit at key `k`.
"""
function _get_name_hierarchy(names)
    d = Dict()
    for name in names
        if name isa Pair
            d[name.first] = push!(get(d, name.first, Set()), name.second)
        else
            get!(d, name, Set())
        end
    end
    return Dict( k => (isnothing(v) ? nothing : _get_name_hierarchy(v)) for (k, v) in d)
end
get_name_hierarchy(events) =
    _get_name_hierarchy([c for (_, c, _) in events])

"""Look up a (potentially nested) address in the name hierarchy."""
hierarchy_lookup(h, addr) = h[addr]
hierarchy_lookup(h, p::Pair) = hierarchy_lookup(h[p.first], p.second)