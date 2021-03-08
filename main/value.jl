# TODO: docstrings; remove old code

##############
# Core Types #
##############
abstract type Value end
abstract type GenericValue <: Value end
abstract type PrimitiveValue{Target} <: Value end
struct CompositeValue{T} <: Value
    vals::T
    CompositeValue(vals::T) where {T <: Union{
        Tuple{Vararg{<:Value}},
        NamedTuple{<:Any, <:Tuple{Vararg{<:Value}}}
    }} = new{T}(vals)
end
IndexedValues(t) = CompositeValue(Tuple(t))
NamedValues(t) = CompositeValue(NamedTuple(t))
Base.pairs(v::CompositeValue) = Base.pairs(v.vals)
Base.keys(v::CompositeValue) = Base.keys(v.vals)
Base.values(v::CompositeValue) = Base.values(v.vals)
Base.length(v::CompositeValue) = Base.length(v.vals)

###############
# Value Types #
###############

"""
    struct Binary <: GenericValue end

Abstract Value representing a signal that is either `0` or `1` (`true` or `false`)
at each moment in time.
"""
struct Binary <: GenericValue end
implement(::Binary, ::Spiking) = SpikeWire()

"""
    FiniteDomainValue(n)

An `AbstractValue` with a finite domain of size `n`.
"""
struct FiniteDomainValue <: GenericValue
    n::UInt
end

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
target(::Type{SpikingCategoricalValue}) = Spiking()
abstract(v::SpikingCategoricalValue) = FiniteDomainValue(v.n)
# performance TODO: can we avoid explicitely constructing a tuple?
implement(v::SpikingCategoricalValue, ::Spiking) = CompositeValue(Tuple(SpikeWire() for _=1:n))

# abstract type Value end

# abstract type Values{VT} <: VT end

# """
#     AbstractValue

# A value or values in an information-processing circuit,
# not tied to any particular hardware target nor encoding scheme.
# """
# abstract type AbstractValue <: Value end

# """
#     AbstractValues <: AbstractValue

# A value consisting of multiple values.
# Iteration yields the values it comprises.
# """
# abstract type AbstractValues <: AbstractValue end

# """
#     named_values(v::Values)

# An iterator over (name, value) pairs for the underlying values.
# By default, `name` is the index of the value.
# """
# named_values(v::Values) = enumerate(v)

# """
#     FiniteDomainValue(n)

# An `AbstractValue` with a finite domain of size `n`.
# """
# struct FiniteDomainValue <: AbstractValue
#     domain_size::UInt
# end

# # struct FinitePrecisionRealValue <: AbstractValue end
# # struct UnbiasedRealValueEstimate <: AbstractValue end

# """
#     ConcreteValue

# A concrete representation of a value or values in an information-processing circuit,
# specialized to a hardware target and an encoding scheme.
# """
# abstract type ConcreteValue <: Value end

# # TODO: docstring
# struct ConcreteValues{Itr}
#     itr::Itr
# end
# iterate(v::ConcreteValues) = iterate(v.itr)
# iterate(v::ConcreteValues, s) = iterate(v.itr, s)

# # TODO: docstring
# struct Wire <: ConcreteValue end
# target(::Type{Wire}) = Spiking()

# """
#     target(::Type{<:ConcreteValue})::Target

# The hardware/software target this type of concrete value is specialized to.
# """
# target(::Type{<:ConcreteValue})::Target = error("Not implemented.")

# """
#     abstract_value(::ConcreteValue)::AbstractValue

# The abstract value that this concrete value implements.
# """
# # TODO: do we also allow this function to return `nothing`?
# abstract_value(::ConcreteValue) = error("Not implemented.")

# """
#     wires(::ConcreteValue)::ConcreteValues

# Returns a `ConcreteValues` of `Wire`s which can transmit the value.
# """
# wires(::ConcreteValue) = error("Not implemented.")

# """
#     CategoricalValue(n)

# A value with a finite domain of size `n`, represented using `n` wires.
# The value `i` is transmitted when the `i`th of the `n` wires spikes.
# """
# struct CategoricalValue <: ConcreteValue
#     domain_size::UInt
# end
# target(::Type{CategoricalValue}) = Spiking()
# abstract_value(v::CategoricalValue) = FiniteDomainValue(v.domain_size)
# wires(c::CategoricalValue) = ConcreteValues((Wire() for _=1:c.domain_size))

# # struct Float64Val <: ConcreteValue end
# # target(::Type{ConcreteValue}) = Spiking()
# # abstract_value(::Float64Val) = FinitePrecisionRealValue()