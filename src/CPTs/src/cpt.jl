abstract type CPT{NumParents} <: Gen.Distribution{Int} end

# TODO: it turns out we can have 0-dimensional arrays, so I think we shouldn't need an explicit ZeroParentCPT subtype!
# TODO: move to using that!
struct ZeroParentCPT <: CPT{0}
    dist::Categorical
end

CPT(dist::Categorical) = ZeroParentCPT(dist)
CPT(dist::Vector{<:Real}) = ZeroParentCPT(Categorical(dist))

Base.getindex(c::ZeroParentCPT) = c.dist
num_inputs(::ZeroParentCPT) = 0
input_ncategories(::ZeroParentCPT) = ()
Distributions.ncategories(c::ZeroParentCPT) = ncategories(c.dist)

struct CPTWithParents{NumParents} <: CPT{NumParents}
    dists::Array{<:Categorical, NumParents}
    num_output_categories::Int

    function CPTWithParents(dists::Array{<:Categorical, NumParents}) where {NumParents}
        @assert !isempty(dists) "Not expecting empty `dists` array when constructing CPT!"
        @assert NumParents > 0 "Use `ZeroParentCPT` if there are no parents!"
        num_cats = ncategories(first(dists))
        @assert all(ncategories(c) === num_cats for c in dists)
        return new{NumParents}(dists, num_cats)
    end
end

CPT(dists::Array{<:Categorical}) = CPTWithParents(dists)
CPT(dists::Array{<:Vector{<:Real}}) = CPTWithParents(map(Categorical, dists))

Base.getindex(c::CPTWithParents, vals...) = c.dists[vals...]
num_inputs(::CPTWithParents{n}) where {n} = n
input_ncategories(c::CPTWithParents) = size(c.dists)
Distributions.ncategories(c::CPTWithParents) = c.num_output_categories

assmts(c::CPTWithParents) = CartesianIndices(input_ncategories(c))

Gen.logpdf(cpt::CPT, val, args...) = log(probs(cpt[args...])[val])
Gen.random(cpt::CPT, args...) = rand(cpt[args...])
Gen.is_discrete(::CPT) = true
(c::CPT)(args...) = Gen.random(c, args...)
Gen.has_output_grad(::CPT) = false
Gen.has_argument_grads(::CPT) = (true,)