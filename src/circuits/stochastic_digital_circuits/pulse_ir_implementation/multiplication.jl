# Why is this a concrete primitive?  Change this?
struct PulseNonnegativeRealMultiplier{M, T} <: ConcretePulseIRPrimitive
    multiplier::M
    inputs::Tuple{Vararg{<:Value}}
    ti::T
    function PulseNonnegativeRealMultiplier(
        multiplier::M, inputs::Tuple{Vararg{<:Value}}, ti::T
    ) where {M, T}
        @assert has_abstract_of_type(multiplier, PulseIR.ConcreteSpikeCountMultiplier)
        @assert all(map(consists_of_spikecount_reals, inputs)) "Inputs does not consist of spikecount reals! -- $inputs"
        @assert has_abstract_of_type(ti, PulseIR.ConcreteThresholdedIndicator)
        return new{M, T}(multiplier, inputs, ti)
    end
end

# TODO: Can I improve this constructor?
function PulseNonnegativeRealMultiplier(
    inputs::Tuple{Vararg{<:Value}},
    MultiplierConstructor, # (indenoms, outdenoms) -> multiplier
    out_count_denominator::Real,
    TIConstructor # threshold -> ti
)
    leafvals = leaf_values(inputs)
    denominators = map(leafvals) do v; v.count_val.denominator; end |> Tuple
    ti = TIConstructor(length(leafvals))
    return PulseNonnegativeRealMultiplier(
        MultiplierConstructor(denominators, out_count_denominator),
        inputs,
        ti
    )
end

Circuits.abstract(m::PulseNonnegativeRealMultiplier) =
    NonnegativeRealMultiplier(m.inputs)
Circuits.inputs(m::PulseNonnegativeRealMultiplier) = IndexedValues(m.inputs)
Circuits.outputs(m::PulseNonnegativeRealMultiplier) = NamedValues(
    :out => IndicatedSpikeCountReal(UnbiasedSpikeCountReal(
        abstract_to_type(m.multiplier, PulseIR.SpikeCountMultiplier).output_count_denominator
    ))
)
    
# TODO: we could be a bit less restrictive, and e.g. support 
# UnbiasedSpikeCountReal which are not indicated.
# But for now, I don't think it's necessary.
consists_of_spikecount_reals(value) =
    value isa IndicatedSpikeCountReal{UnbiasedSpikeCountReal} ||
    (
        value isa ProductNonnegativeReal &&
        all(map(consists_of_spikecount_reals, value.factors))
    )

# Get a list the leaf SingleNonnegativeReals
leaf_values(values::Union{<:Tuple, <:NamedTuple}) = flatten(map(leaf_values, values))
leaf_values(v::IndicatedSpikeCountReal{UnbiasedSpikeCountReal}) = [v]
leaf_values(v::ProductNonnegativeReal) = leaf_values(v.factors)

# Get the addresses of the leafs
prepend(::Nothing, list) = list
prepend(prefix, list) = (prefix => item for item in list)
leaf_value_addresses(values, prefix=nothing) =
    flatten([
        prepend(prefix, leaf_value_addresses(value, i))
        for (i, value) in pairs(values)
    ])
leaf_value_addresses(value::ProductNonnegativeReal, prefix) =
    leaf_value_addresses(value.factors, prefix)
leaf_value_addresses(
    ::IndicatedSpikeCountReal{UnbiasedSpikeCountReal},
    prefix
) = [prefix]
flatten(x) = (collect ∘ Iterators.flatten)(x)

Circuits.implement(m::PulseNonnegativeRealMultiplier, ::Spiking) =
    CompositeComponent(
        implement_deep(inputs(m), Spiking()), implement_deep(outputs(m), Spiking()),
        (
            multiplier=m.multiplier,
            ti=m.ti
        ),
        (
            (
                Input(Circuits.nest(addr, :ind)) => CompIn(:ti, :in)
                for addr in leaf_value_addresses(m.inputs)
            )...,
            (
                Input(Circuits.nest(addr, :count)) => CompIn(:multiplier, :counts => i)
                for (i, addr) in enumerate(leaf_value_addresses(m.inputs))
            )...,
            CompOut(:ti, :out) => CompIn(:multiplier, :ind),
            CompOut(:multiplier, :ind) => Output(:out => :ind),
            CompOut(:multiplier, :count) => Output(:out => :count)
        ),
        m
    )