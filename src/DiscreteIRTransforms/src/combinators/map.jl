function ensure_product_inputs(arg_domains)
    for dom in arg_domains
        @assert dom isa ProductDomain "$dom is used where we need a ProductDomain"
    end
end

function to_labeled_cpts(m::Gen.Map{T, U}, arg_domains) where {T, U}
    ensure_product_inputs(arg_domains)
    maplength = length(first(arg_domains).sub_domains)
    @assert all(length(dom.sub_domains) == maplength for dom in arg_domains)

    compiled_kernels = [
        to_labeled_cpts(m.kernel, domain_assmt)
        for domain_assmt in zip((dom.sub_domains for dom in arg_domains)...)
    ]

    return ApplyCombinator.Apply{T, apply_tracetype(compiled_kernels)}(compiled_kernels)
end

function get_ret_domain(m::Gen.Map, arg_domains)
    ensure_product_inputs(arg_domains)
    return ProductDomain([
        get_ret_domain(m.kernel, domain_assmt)
        for domain_assmt in zip((dom.sub_domains for dom in arg_domains)...)
    ])
end
is_cpts(m::Gen.Map) = is_cpts(m.kernel)

# TODO: to_indexed_cpts
# (we don't need it now since currently we convert every Map to Apply in 
# to_labeled_cpts; in the future we might want `Map`)

# TODO: remove args which have domain of size 1





# function unpack_map_domains(domains)
#     for (i, dom) in enumerate(domains)
#         @assert dom isa ProductDomain
#         first_subdom, other_subdoms = Iterators.peel(dom.sub_domains)
#         @assert all(unordered_isequal(d, first_subdom) for d in other_subdoms) "Input $i to Map has different input domains at different positions!  Input domains: $(dom.sub_domains)."
#     end

#     num_calls = length(first(domains).sub_domains)
#     single_call_domains = [first(d.sub_domains) for d in domains]
#     return (single_call_domains, num_calls)
# end

# function get_ret_domain(m::Gen.Map, arg_domains)
#     (single_call_domains, n_repetitions) = unpack_map_domains(arg_domains)
#     kernel_ret_dom = get_ret_domain(m.kernel, single_call_domains)
#     return ProductDomain(
#         Iterators.repeated(kernel_ret_dom, n_repetitions) |> collect
#     )
# end

# function to_labeled_cpts(m::Gen.Map, arg_domains)
#     (single_call_domains, _) = unpack_map_domains(arg_domains)
#     return Gen.Map(to_labeled_cpts(m.kernel, single_call_domains))
# end

# function to_indexed_cpts(m::Map, arg_domains)
#     (single_call_domains, num_calls) = unpack_map_domains(arg_domains)
#     (new_kernel, bijs, retbij) = to_indexed_cpts(m.kernel, single_call_domains)

#     return (
#         Gen.Map(new_kernel),
#         Dict(i => bijs for i=1:num_calls),
#         [retbij for _=1:num_calls]
#     )
# end
