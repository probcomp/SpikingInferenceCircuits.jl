"""
    ToAssmts{n} <: GenericComponent
    ToAssmts((s1, s2, ..., sn))

A component which recieves `n` finite domain values with sizes `s1, s2, ..., sn`,
and outputs a single finite domain value with domain size `s1 × s2 × ... × sn` corresponding
an assignment of one value to each input variable.  (Order of the output
given by converting Julia's `CartesianIndices → LinearIndices`.)
"""
struct ToAssmts{n} <: GenericComponent
    size::NTuple{n, Int}
end
Circuits.inputs(a::ToAssmts) = IndexedValues(FiniteDomainValue(n) for n in a.size)
Circuits.outputs(a::ToAssmts) = CompositeValue((out=FiniteDomainValue(prod(a.size)),))