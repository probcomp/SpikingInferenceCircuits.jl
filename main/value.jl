# TODO: docstrings; remove old code

##############
# Core Types #
##############

"""
    abstract type Value end

A "wire" in a circuit diagram, which can transmit a value of a certain type.
"""
abstract type Value end

"""
    abstract type GenericValue <: Value end

A `Value` which is not a `PrimitiveValue` nor `CompositeValue`.
"""
abstract type GenericValue <: Value end

"""
    abstract type PrimitiveValue{T} <: Value end

A primitive value for `T  <: Target`.  Ie. the simulator for `T`
can directly use this type of value.
"""
abstract type PrimitiveValue{Target} <: Value end

"""
    implement(::Value, ::Target)

Make progress implementing the value for the target, so that a finite
number of repeated calls to `implement` will yield a value `v` so that `is_implementation_for(v, t)` is true.
"""
implement(::V, ::T) where {V <: Value, T <: Target} = error("No implementation for values of type `$V` defined for target `$T`.")

"""
    is_implementation_for(::Value, ::Target)

Whether the given value is an implementation for the given target (ie. whether it is supported
by the simulator for that target).
"""
is_implementation_for(::PrimitiveValue{<:T}, ::T) where {T <: Target} = true
is_implementation_for(::PrimitiveValue, ::Target) = false
is_implementation_for(::GenericValue, ::Target) = false

"""
    implement_deep(::Value, t::Target)

Implement the value for the target recursively, yielding a value `v` such that
`is_implementation_for(v, t)` is true.
"""
implement_deep(v::PrimitiveValue{U}, t::Target) where {U <: Target} = error("Cannot implement $v, a PrimitiveValue{$u}, for target $t.")
implement_deep(v::PrimitiveValue{<:T}, ::T) where {T <: Target} = v
implement_deep(v::GenericValue, t::Target) = implement_deep(implement(v, t), t)

##################
# CompositeValue #
##################

"""
    CompositeValue <: Value
    CompositeValue(vals::Union{Tuple, NamedTuple}, abstract=nothing)

A value composed from other values.

If constructed with `vals::Tuple`, the sub-values are named `1, ..., length(vals)`;
if constructed with `vals::NamedTuple`, the sub-values are named using the named tuple keys.
`abstract` is the more abstract value which was implemented to yield this composite value.

Sub-values may be accessed using `Base.getindex`, so `v[name]` yields the subvalue with that name
in `v::CompositeValue`.  To access a sub-value nested within several layers of `CompositeValue`s,
one may use `v[x1 => (x2 => ... => (x_n))]`, which equals `v[x1][x2][...][x_n]`.

`Base.pairs(::CompositeComponent)` iterates over `(val_name, sub_value)` pairs,
`Base.keys` gives an iterator over the names, and `Base.values` gives an iterator over the sub-values.
"""
struct CompositeValue{T} <: Value
    vals::T
    abstract::Union{Value, Nothing}
    CompositeValue(vals::T, abstract::Union{Value, Nothing}=nothing) where {T <: Union{
            Tuple{Vararg{<:Value}},
            NamedTuple{<:Any, <:Tuple{Vararg{<:Value}}}
        }
    } = new{T}(vals, abstract)
end

"""
    IndexedValues(t)

Given iterator `t` over values,
a `CompositeValue` with sub-value names, `1, ..., length(t)` and values
given by iterating through `t`.
"""
IndexedValues(t) = CompositeValue(Tuple(t))

"""
    NamedValues(t)

Given iterator `t` over `(name::Symbol, value::Value)` pairs,
a `CompositeValue` with the given values at the given names.
"""
NamedValues(t) = CompositeValue(NamedTuple(t))

Base.pairs(v::CompositeValue) = Base.pairs(v.vals)
Base.keys(v::CompositeValue) = Base.keys(v.vals)
Base.values(v::CompositeValue) = Base.values(v.vals)
Base.length(v::CompositeValue) = Base.length(v.vals)

# TODO: performance
length_deep(v::CompositeValue) = sum(length_deep(sv) for sv in values(v))
length_deep(::PrimitiveValue) = 1
length_deep(::GenericValue) = 1

"""
    keys_deep(v::CompositeValue)

An iterator over all the nested value names (nesting until reaching
a non-composite value).

### Example
```julia
c = CompositeValue((
    a = CompositeValue((SpikeWire(), SpikeWire())).
    b = CompositeValue((k=SomeGenericValue(),)),
    c = SomeGenericValue()
))
collect(keys_deep)(c) # == [:a => 1, :a => 2, :b => :k, :c]
```
"""
keys_deep(v::CompositeValue) = Iterators.flatten((
    val isa CompositeValue ? (k => subkey for subkey in keys_deep(val)) : (k,)
    for (k, val) in Base.pairs(v)
))

abstract(v::CompositeValue) = v.abstract

Base.getindex(cv::CompositeValue, k) = cv.vals[k]
Base.getindex(cv::CompositeValue, p::Pair) = cv.vals[p.first][p.second]

# TODO: faster implementations for `is_implementation_for`, `implement_deep`?
is_implementation_for(v::CompositeValue, t::Target) = all(is_implementation_for(val, t) for val in values(v))
implement_deep(v::CompositeValue, t::Target) =
    if is_implementation_for(v, t)
        v
    else
        CompositeValue(map(val -> implement_deep(val, t), v.vals), v)
    end

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
implement(v::SpikingCategoricalValue, ::Spiking) = CompositeValue(Tuple(SpikeWire() for _=1:v.n), v)

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
abstract(::BinarySamplesUnbiasedPositiveReal) = UnbiasedPositiveReal()
implement(s::BinarySamplesUnbiasedPositiveReal, ::Target) = IndexedValues(FiniteDomainValue(2) for _=1:s.num_samples)
