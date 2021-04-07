function unpack_map_domains(domains)
    for dom in domains
        @assert dom isa ProductDomain
        @assert all(d == first(dom.sub_domains) for d in dom.sub_domains[2:end])
    end

    num_calls = length(first(arg_domains).sub_domains)
    single_call_domains = [first(d.sub_domains) for d in arg_domains]
    return (single_call_domains, num_calls)
end

function get_ret_domain(m::Gen.Map, arg_domains)
    (single_call_domains, n_repetitions) = unpack_map_domains(arg_domains)
    kernel_ret_dom = get_ret_dom(m.kernel, single_call_domains)
    return ProductDomain(
        Iterators.repeated(kernel_ret_dom, n_repetitions) |> collect
    )
end

function to_labeled_cpts(m::Gen.Map, arg_domains)
    (single_call_domains, _) = unpack_map_domains(arg_domains)
    return Gen.Map(to_labeled_cpts(m.kernel, single_call_domains))
end

function to_indexed_cpts(m::Map, arg_domains)
    (single_call_domains, num_calls) = unpack_map_domains(arg_domains)
    (new_kernel, bijs) = to_indexed_cpts(m.kernel, single_call_domains)

    return (
        Gen.Map(new_kernel),
        Dict(i => bijs for i=1:num_calls)
    )
end
