#=
Utilities for dealing with values, domains, etc.,
when constructing circuits which implement Generative Function operations.
=#

### utils: get_selected, has_selected ###

"""
    get_selected(value::Value, sel::Selection)

Filters a nested value to only include selected nodes.
Returns a value which filters the given `value` 
so that the only leaf nodes are those at addresses selected in `sel`.
"""
get_selected(val::FiniteDomainValue, ::AllSelection) = val
get_selected(::FiniteDomainValue, ::EmptySelection) = error()
function get_selected(val::FiniteDomainValue, s::ComplementSelection)
    @assert s.complement == EmptySelection() "complement was $(s.complement)"
    val
end
get_selected(val::CompositeValue, sel::Selection) = NamedValues((
        addr => get_selected(subval, sel[addr])
        for (addr, subval) in pairs(val)
        if has_selected(subval, sel[addr])
    )...)

function has_selected(::FiniteDomainValue, s::ComplementSelection)
    if isempty(s.complement)
        return true
    elseif s.complement == AllSelection()
        return false
    else
        error("unexpected")
    end
end
has_selected(_, ::AllSelection) = true
has_selected(_, ::EmptySelection) = false
has_selected(v::CompositeValue, s::Selection) = _has_selected_nontrivial(v, s)
has_selected(v::CompositeValue, s::AllSelection) = _has_selected_nontrivial(v, s)
_has_selected_nontrivial(v, s) = any(
        has_selected(sv, s[a]) for (a, sv) in pairs(v)
    )

### Domain types ###
# Note that this type of domain is different from `DiscreteIRTransforms.Domain`.
# (A FiniteDomain(n) is equivalent to an DiscreteIRTransforms.EnumeratedDomain(1:n).)

abstract type Domain end
struct FiniteDomain <: Domain
    n::Int
end
struct IndexedProductDomain{T} <: Domain
    subdomains::T
end

to_value(f::FiniteDomain) = FiniteDomainValue(f.n)
to_value(p::IndexedProductDomain) = CompositeValue(Tuple(to_value(v) for v in p.subdomains))