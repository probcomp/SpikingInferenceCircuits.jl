"""
    NonnegativeRealMultiplier <: GenericComponent
    NonnegativeRealMultiplier(inputs_values)

Multiplies one or more `NonnegativeReal`s together.
`input_values` is the tuple of Values which carry the values to multiply.
"""
struct NonnegativeRealMultiplier <: GenericComponent
    inputs::Tuple{Vararg{<:Value}}
end
Circuits.inputs(r::NonnegativeRealMultiplier) = IndexedValues(r.inputs)
Circuits.outputs(r::NonnegativeRealMultiplier) = NamedValues(:out => NonnegativeReal())
