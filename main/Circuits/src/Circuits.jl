module Circuits

using Distributions: Categorical, ncategories, probs

# returns a collection of the same top-level type mapping `name => name`
names(t::Tuple) = Tuple(1:length(t))
names(n::NamedTuple) = (;(k=>k for k in keys(n))...)

no_impl_error(::V, ::T) where {V, T} = error("No implementation for type `$V` defined for target `$T`.")

include("target.jl")
include("value.jl")
include("component.jl")

"""
    can_implement(c::Component, t::Target)
    can_implement(v::Value, t::Target)

Whether the given component/value can be implemented in this target.
"""
can_implement(::K, ::T) where {K <: Union{Component, Value}, T <: Target} = hasmethod(implement, Tuple{K, T})

export Target
export Value, PrimitiveValue, GenericValue, CompositeValue
export Component, PrimitiveComponent, GenericComponent, CompositeComponent
export abstract, target, inputs, outputs, implement, implement_deep, is_implementation_for
export keys_deep, length_deep
export IndexedValues, NamedValues
export ComponentGroup, IndexedComponentGroup, NamedComponentGroup
export NodeName, Input, Output, CompIn, CompOut
export Binary
export can_implement, compiles_to_binary

end