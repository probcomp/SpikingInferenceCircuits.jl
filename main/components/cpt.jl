using Distributions: Categorical

struct ConditionalProbabilityTable{num_in_vars}
    dists::Array{<:Categorical, num_in_vars}
    num_output_categories::Int

    function ConditionalProbabilityTable(dists::Array{<:Categorical, niv}) where {niv}
        @assert !isempty(dists) "Not expecting empty `dists` array when constructing CPT!"
        num_cats = ncategories(first(dists))
        @assert all(ncategories(c) === num_cats for c in dists)
        return ConditionalProbabilityTable{niv}(dists, num_cats)
    end
end
const CPT = ConditionalProbabilityTable

Base.getindex(c::CPT, vals...) = c.dists[vals]
num_inputs(::CPT{n}) where {n} = n
input_ncategories(c::CPT) = size(c.dists)
Distributions.ncategories(c::CPT) = c.num_output_categories