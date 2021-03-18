module Circuits

using Distributions: Categorical, ncategories, probs

# returns a collection of the same top-level type mapping `name => name`
names(t::Tuple) = Tuple(1:length(t))
names(n::NamedTuple) = (;(k=>k for k in keys(n))...)

no_impl_error(::V, ::T) where {V, T} = error("No implementation for type `$V` defined for target `$T`.")

include("target.jl")
include("value.jl")
include("component.jl")

export Target
export Value, PrimitiveValue, GenericValue, CompositeValue
export Component, PrimitiveComponent, GenericComponent, CompositeComponent
export abstract, target, inputs, outputs, implement, implement_deep, is_implementation_for
export keys_deep, length_deep
export IndexedValues, NamedValues
export ComponentGroup, IndexedComponentGroup, NamedComponentGroup
export NodeName, Input, Output, CompIn, CompOut
export Binary

end