"""
    Theta <: GenericComponent
    Theta(n_possibilities::Int)

Given `n` `NonnegativeReal`s `[v₁, ..., vₙ]`, normalizes
the vector to a probability vector `[p₁, ..., pₙ]` and
outputs a sample from the resulting distribution.
"""
struct Theta <: GenericComponent
    n_possibilities::Int
end
Circuits.inputs(t::Theta) = IndexedValues(NonnegativeReal() for _=1:t.n_possibilities)
Circuits.outputs(t::Theta) = NamedValues(:val => FiniteDomainValue(t.n_possibilities))