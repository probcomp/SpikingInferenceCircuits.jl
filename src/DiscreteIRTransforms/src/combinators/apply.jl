function to_labeled_cpts(a::ApplyCombinator.Apply{T, U}, arg_domains) where {T, U}
    @assert all(dom isa ProductDomain for dom in arg_domains)
    maplength = length(first(arg_domains).sub_domains)
    @assert all(length(dom.sub_domains) == maplength for dom in arg_domains)

    compiled_kernels = [
        to_labeled_cpts(kernel, domain_assmt)
        for (kernel, domain_assmt) in zip(
            a.kernels,
            zip((dom.sub_domains for dom in arg_domains)...)
        )
    ]

    return ApplyCombinator.Apply{T, apply_tracetype(compiled_kernels)}(compiled_kernels)
end

get_ret_domain(a::ApplyCombinator.Apply, arg_domains) =
    ProductDomain([
        get_ret_domain(kernel, domain_assmt)
        for (kernel, domain_assmt) in zip(
            a.kernels,
            zip((dom.sub_domains for dom in arg_domains)...)
        )
    ])

function to_indexed_cpts(a::ApplyCombinator.Apply, arg_domains)
    compiled_kernels_and_mappings = [
        to_indexed_cpts(kernel, domain_assmt)
        for (kernel, domain_assmt) in zip(
            a.kernels,
            zip((dom.sub_domains for dom in arg_domains)...)
        )
    ]
    compiled_kernels = map(first, compiled_kernels_and_mappings)
    bijs = map(x -> x[2], compiled_kernels_and_mappings)
    retbijs = map(x -> x[3], compiled_kernels_and_mappings)

    return (
        ApplyCombinator.Apply{
            Union{(Gen.get_return_type(k) for k in compiled_kernels)...},
            apply_tracetype(compiled_kernels)
        }(compiled_kernels),
        Dict(i => bij for (i, bij) in enumerate(bijs)),
        retbijs
    )
end

is_cpts(a::ApplyCombinator.Apply) = all(is_cpts(k) for k in a.kernels)

with_constant_inputs_at_indices(a::ApplyCombinator.Apply, idx_val_pairs) =
    ApplyCombinator.Apply{
        # TODO
    }(

    )