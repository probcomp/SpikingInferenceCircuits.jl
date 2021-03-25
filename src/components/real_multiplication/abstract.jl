"""
    PositiveRealMultiplier <: GenericComponent
    PositiveRealMultiplier(n)

Multiplies `n` `PositiveReal`s together.
"""
struct PositiveRealMultiplier <: GenericComponent
    n_inputs::Int
end
Circuits.inputs(r::PositiveRealMultiplier) = IndexedValues(PositiveReal() for _=1:r.n_inputs)
Circuits.outputs(r::PositiveRealMultiplier) = NamedValues(:out => PositiveReal())