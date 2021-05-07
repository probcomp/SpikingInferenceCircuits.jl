"""
    NonnegativeRealMultiplier <: GenericComponent
    NonnegativeRealMultiplier(n)

Multiplies `n` `NonnegativeReal`s together.
"""
struct NonnegativeRealMultiplier <: GenericComponent
    n_inputs::Int
end
Circuits.inputs(r::NonnegativeRealMultiplier) = IndexedValues(NonnegativeReal() for _=1:r.n_inputs)
Circuits.outputs(r::NonnegativeRealMultiplier) = NamedValues(:out => NonnegativeReal())