struct CPT{num_inputs, num_assmts, ArrType}
    input_domain_sizes::NTuple{num_inputs, UInt}
    output_domain_size::UInt,
    dists::ArrType

    function CPT(cpt::AbstractArray{Categorical, n})
        output_domain_size = ncategories(first(cpt))
        input_domain_sizes = size(cpt)

        if !isempty(cpt)
            @assert all(ncategories(dist) == output_domain_size for dist in cpt) "Output support size is not constant!"
        end

        new{length(input_domain_sizes), prod(input_domain_sizes)}(
            input_domain_sizes, output_domain_size, cpt
        )
end