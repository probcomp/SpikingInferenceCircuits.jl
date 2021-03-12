module Circuits

using Distributions: Categorical, ncategories, probs

include("target.jl")
include("value.jl")
include("component.jl")

export Target
export Value, PrimitiveValue, GenericValue, CompositeValue
export Component, PrimitiveComponent, GenericComponent, CompositeComponent
export abstract, target, inputs, outputs, implement, implement_deep, is_implementation_for
export keys_deep
export IndexedValues, NamedValues
export ComponentGroup, IndexedComponentGroup, NamedComponentGroup
export NodeName, Input, Output, CompIn, CompOut

end