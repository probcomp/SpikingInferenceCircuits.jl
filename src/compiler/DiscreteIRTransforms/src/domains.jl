### Domain type ###
abstract type Domain end
"""Domain represented as a list of possible values."""
struct EnumeratedDomain{V} <: Domain
    vals::V
end
vals(d::EnumeratedDomain) = d.vals

"""
Product set of all tuples/vectors containing one object each subdomain.
(`vector_valued` is true if the domain contains vectors, false otherwise.)
"""
struct ProductDomain{T <: Union{
    Tuple{Vararg{<:Domain}},
    Vector{<:Domain}
}} <: Domain
    sub_domains::T
    vector_valued::Bool
end
vals(d::ProductDomain) = (
        d.vector_valued ? collect(v) : v
        for v in Iterators.product(d.sub_domains)
    )

Base.iterate(d::Domain) = Base.iterate(vals(d))
Base.iterate(d::Domain, s) = Base.iterate(vals(d), s)
Base.length(d::Domain) = length(vals(d))
Base.:(==)(a::Domain, b::Domain) = all(x == y for (x, y) in zip(a, b))
Base.hash(d::Domain, h::UInt) = hash(collect(vals(d)), h)

### Indicator in Generative Function that we should use a ProductDomain ###
# struct IndepDomainVector{V} <: AbstractVector
#     v::V
# end
# Base.size(v::IndepDomainVector) = Base.size(v)
# Base.getindex(v::IndepDomainVector, )

# struct IndepDomainTuple{T <: Tuple}
#     t::T
# end

# is_indep_domain_value(::IndepDomainVector) = true
# is_indep_domain_value(::IndepDomainTuple) = true
# is_indep_domain_value(_) = false
