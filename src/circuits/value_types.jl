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
    PositiveReal <: GenericValue

A positive real number.
"""
struct PositiveReal <: GenericValue end

"""
    SpikeRateReal <: GenericValue
    SpikeRateReal(reference_rate)

A nonnegative real number encoded via the rate of spiking in a wire.  If the spike rate is `r`,
the encoded real is `r/reference_rate`.
"""
struct SpikeRateReal <: GenericValue
    reference_rate::Float64
end
Circuits.abstract(::SpikeRateReal) = PositiveReal()
Circuits.target(::SpikeRateReal) = Spiking()
Circuits.implement(::SpikeRateReal, ::Spiking) = SpikeWire()

####

####

"""
    UnbiasedPositiveReal <: GenericValue

An unbiased estimate of a positive real number.
"""
struct UnbiasedPositiveReal <: GenericValue end

"""
    BinarySamplesUnbiasedPositiveReal <: GenericValue
    BinarySamplesUnbiasedPositiveReal(n)

An abstract implementation of a `UnbiasedPositiveReal`.
Sends `n` binary "sample" values (as `FiniteDomainValue(2)`s).
If `n1` samples of `1` and `n2` samples of `2` are sent,
the transmitted value is `n1/(1 + n2)`.

If the probability of any sample being `1` is `p/(p + q)`, then in expectation
the transmitted value is `p/q` (so this is an unbiased estimate of `p/q`).
"""
struct BinarySamplesUnbiasedPositiveReal <: GenericValue
    num_samples::UInt
end
Circuits.abstract(::BinarySamplesUnbiasedPositiveReal) = UnbiasedPositiveReal()
Circuits.implement(s::BinarySamplesUnbiasedPositiveReal, ::Target) = IndexedValues(FiniteDomainValue(2) for _=1:s.num_samples)