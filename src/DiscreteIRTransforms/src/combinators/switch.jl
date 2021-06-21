function get_ret_domain(s::Gen.Switch, arg_domains)
    # first domain should be branch selection
    @assert arg_domains[1] == EnumeratedDomain(1:length(arg_domains[1]))

    ret_domains = [get_ret_domain(branch, arg_domains[2:end]) for branch in s.branches]
    @assert all(d == first(ret_domains) for d in ret_domains[2:end]) "Different branches had different return domains! Domains for each branch: $ret_domains"
    return first(ret_domains)
end

function to_labeled_cpts(s::Switch, arg_domains)
    return Switch((to_labeled_cpts(b, arg_domains[2:end]) for b in s.branches)...)
end

unzip(v) = Tuple(map(x -> x[i], v) for i=1:length(first(v)))
function to_indexed_cpts(s::Switch, arg_domains)
    (indexed_fns, bijs, retbijs) = unzip([to_indexed_cpts(b, arg_domains[2:end]) for b in s.branches])
    return (
        Switch((indexed_fns)...),
        Dict(i => b for (i, b) in enumerate(bijs)),
        only(unique(retbijs))
    )
end

is_cpts(s::Switch) = all(is_cpts(b) for b in s.branches)