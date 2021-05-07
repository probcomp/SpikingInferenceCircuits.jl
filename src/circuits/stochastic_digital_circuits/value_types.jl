###############
# Value Types #
###############

"""
    FiniteDomainValue(n)

An `Value` with a finite domain of size `n`.
"""
struct FiniteDomainValue <: GenericValue
    n::UInt
end
Circuits.implement(v::FiniteDomainValue, ::Spiking) = SpikingCategoricalValue(v.n)

"""
    SpikingCategoricalValue(n)

A value with a finite domain of size `n`, represented using `n` wires.
The value `i` is transmitted when the `i`th of the `n` wires spikes.
"""
struct SpikingCategoricalValue <: GenericValue
    n::UInt
end
SpikingCategoricalValue(f::FiniteDomainValue) = SpikingCategoricalValue(f.n)
FiniteDomainValue(f::SpikingCategoricalValue) = FiniteDomainValue(f.n)
Circuits.target(::SpikingCategoricalValue) = Spiking()
Circuits.abstract(v::SpikingCategoricalValue) = FiniteDomainValue(v.n)
# performance TODO: can we avoid explicitely constructing a tuple?
Circuits.implement(v::SpikingCategoricalValue, ::Spiking) = CompositeValue(Tuple(SpikeWire() for _=1:v.n), v)

"""
    NonnegativeReal <: GenericValue

A positive real number.
"""
struct NonnegativeReal <: GenericValue end

"""
A nonnegative real transmitted as a single value (ie. not via sub-values which should be added or multiplied together).
"""
struct SingleNonnegativeReal <: GenericValue end
Circuits.abstract(::SingleNonnegativeReal) = NonnegativeReal()

"""
A NonnegativeReal transmitted by sending `length(factors)`
separate NonnegativeReal values, which should be multiplied together
to obtain the represented value.
"""
struct ProductNonnegativeReal <: GenericValue
    factors
    function ProductNonnegativeReal(factors::Union{<:Tuple, <:NamedTuple})
        @assert all(has_abstract_of_type(factor, NonnegativeReal) for factor in factors)
        return new(factors)
    end
end
Circuits.abstract(::ProductNonnegativeReal) = NonnegativeReal()
Circuits.implement(r::ProductNonnegativeReal, ::Target) = CompositeValue(r.factors, r)

"""
    UnbiasedSpikeCountReal(denominator)

A nonnegative real number `r` encoded via `C` spikes in a wire such that
`E[C/denominator] = r`.
"""
struct UnbiasedSpikeCountReal <: GenericValue
    denominator::Float64
end
Circuits.abstract(::UnbiasedSpikeCountReal) = SingleNonnegativeReal()
Circuits.target(::UnbiasedSpikeCountReal) = Spiking()
Circuits.implement(::UnbiasedSpikeCountReal, ::Spiking) = SpikeWire()