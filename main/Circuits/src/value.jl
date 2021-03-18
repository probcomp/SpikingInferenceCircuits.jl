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
implement(v::Value, t::Target) = no_impl_error(v, t)
implement(v::PrimitiveValue{T1}, t::T2) where {T1 <: Target, T2 <: Target} =
    if T1 <: T2
        v
    else
        error("Cannot implement $v, a PrimitiveValue{$T1}, for target $t.")
    end

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
implement_deep(v::PrimitiveValue, t::Target) = implement(v, t)
implement_deep(v::GenericValue, t::Target) = implement_deep(implement(v, t), t)

"""
    abstract(v::Value)

Returns the value which was `implement`ed to produce `v` if it exists and is available;
otherwise returns `nothing`.
"""
abstract(::Value) = nothing

"""
    target(v::Value)

If `v` is a _concrete value_ which has 1 implementation for 1 target,
return the `Target` `v` can be implemented for.  Else, returns `nothing`.
"""
target(::Value) = nothing

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
struct CompositeValue <: Value# {T} <: Value
    vals #::T
    abstract::Union{Value, Nothing}
    # CompositeValue(v::T, abst=nothing) where {T <: tup_or_namedtup(Value)} = new{basetype(T)}(v, abst)
    # CompositeValue(v::T, args...) where {T <: Tuple{Vararg{<:Value}}} = new{Tuple}(vals, args...)
    # CompositeValue(v::T, args...) where {T <: NamedTuple{<:Any, <:Tuple{Vararg{<:Value}}}} = new{NamedTuple}(vals, args...)
end
CompositeValue(vals) = CompositeValue(vals, nothing)

"""
    IndexedValues(t)

Given iterator `t` over values,
a `CompositeValue` with sub-value names, `1, ..., length(t)` and values
given by iterating through `t`.
"""
IndexedValues(t, abstract=nothing) = CompositeValue(Tuple(t), abstract)

"""
    NamedValues(t...)

Given a vararg `t` of `(name::Symbol, value::Value)` pairs,
a `CompositeValue` with the given values at the given names.
"""
NamedValues(t...) = CompositeValue((;t...))

Base.pairs(v::CompositeValue) = Base.pairs(v.vals)
Base.keys(v::CompositeValue) = Base.keys(v.vals)
Base.values(v::CompositeValue) = Base.values(v.vals)
Base.length(v::CompositeValue) = Base.length(v.vals)

# TODO: performance
length_deep(v::CompositeValue) = reduce(+, length_deep(sv) for sv in values(v); init=0)
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

is_implementation_for(v::CompositeValue, t::Target) = all(is_implementation_for(val, t) for val in values(v))

"""
    implement(v::CompositeValue, t::Target, filter::Function = (x -> true); deep=false)

Make progress implementing `v` for `t`.  All subvalues whose name `n` is such that `filter(n) == true`
will be implemented; by default all subvalues are implemented.

If `deep` is false, each subvalue will be implemented one "level" using `implement`; if `deep` is true,
each subvalue will be fully implemented via `implement_deep`.
"""
implement(v::CompositeValue, t::Target, filter::Function = (x -> true); deep=false) =
    CompositeValue(
        map(names(v.vals), v.vals) do name, val
            if filter(name)
                if deep
                    implement_deep(val, t)
                else
                    implement(val, t)
                end
            else
                val
            end
        end,
        v
    )

"""
    implement(v::CompositeValue, t::Target, names...; deep=false)
    implement_deep(v::CompositeValue, t::Target, names...)

Implement or implement deep all subvalues of `v` whose name is in `names`.
"""
implement(v::CompositeValue, t::Target, names...; kwargs...) = implement(v, t, in(names); kwargs...)
implement_deep(v::CompositeValue, t::Target, names...) = implement(v, t, names...; deep=true)

### Binary ###
"""
    Binary <: GenericValue

A value which at any given time either represents a `1` or `0`.
"""
struct Binary <: GenericValue end

"""
    compiles_to_binary(v::Value, t::Target)

Whether this value can be compiled for this target
into a `CompositeValue` made entirely of `Binary` values
(or values which are implementations of `Binary`).
"""
compiles_to_binary(v::Value, t::Target) = _compiles_to_binary(v, t)
_compiles_to_binary(v::CompositeValue, t::Target) = all(compiles_to_binary(val, t) for val in v.vals)
_compiles_to_binary(v::Value, t::Target) = compiles_to_binary(implement(v, t), t)
_compiles_to_binary(::Binary, ::Target) = true