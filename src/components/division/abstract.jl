"""
    PositiveRealDivider <: GenericComponent
    PositiveRealDivider()

Divides PositiveReal input `:numerator` by PositiveReal input `:denominator`.
"""
struct PositiveRealDivider <: GenericComponent end
Circuits.inputs(::PositiveRealDivider) = NamedValues(:numerator => PositiveReal(), :denominator => PositiveReal())
Circuits.outputs(::PositiveRealDivider) = NamedValues(:out => PositiveReal())