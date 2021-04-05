using Distributions: Categorical, ncategories, probs
import Distributions
import Gen

struct ConditionalProbabilityTable{num_in_vars} <: Gen.Distribution{Int}
    dists::Array{<:Categorical, num_in_vars}
    num_output_categories::Int

    function ConditionalProbabilityTable(dists::Array{<:Categorical, niv}) where {niv}
        @assert !isempty(dists) "Not expecting empty `dists` array when constructing CPT!"
        num_cats = ncategories(first(dists))
        @assert all(ncategories(c) === num_cats for c in dists)
        return new{niv}(dists, num_cats)
    end
end
ConditionalProbabilityTable(dists::Array{<:Vector}) = ConditionalProbabilityTable(map(Categorical, dists))
const CPT = ConditionalProbabilityTable

Base.getindex(c::CPT, vals...) = c.dists[vals...]
num_inputs(::CPT{n}) where {n} = n
input_ncategories(c::CPT) = size(c.dists)
Distributions.ncategories(c::CPT) = c.num_output_categories

assmts(c::CPT) = CartesianIndices(input_ncategories(c))

Gen.logpdf(cpt::CPT, val, args...) = log(probs(cpt[args...])[val])
Gen.random(cpt::CPT, args...) = rand(cpt[args...])
Gen.is_discrete(::CPT) = true
(c::CPT)(args...) = random(c, args...)
Gen.has_output_grad(::CPT) = false
Gen.has_argument_grads(::CPT) = (true,)

get_cpt(cpt::CPT) = cpt