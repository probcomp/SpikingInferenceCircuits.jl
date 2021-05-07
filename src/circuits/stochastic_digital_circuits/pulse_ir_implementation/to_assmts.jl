struct PulseToAssmts{n} <: ConcretePulseIRPrimitive
    size::NTuple{n, Int}
    ti_type::Type # one of ThresholdedIndicator, ConcreteThresholdedIndicator, PoissonThresholdedIndicator, etc.
    ti_params::Tuple # other than `threshold`
end
# PulseToAssmts(size::NTuple{n, Int}, args...) where {n} = PulseToAssmts{n}(size, args...)
PulseToAssmts(ta::ToAssmts, args...) = PulseToAssmts(ta.size, args...)

Circuits.abstract(a::PulseToAssmts) = ToAssmts(a.size)
Circuits.inputs(a::PulseToAssmts) =
    IndexedValues(
        implement(SpikingCategoricalValue(n), Spiking())
        for n in a.size
    )
Circuits.outputs(a::PulseToAssmts) =
    NamedValues(
        :out => implement(
            SpikingCategoricalValue(prod(a.size)),
            Spiking()
        )
    )

Circuits.implement(a::PulseToAssmts, ::Spiking) =
    CompositeComponent(
        inputs(a), outputs(a),
        Tuple(
            a.ti_type(
                length(a.size), a.ti_params...
            ) for _=1:length(outputs(a)[:out])
        ),
        Iterators.flatten(
            (
                (
                    Input(i => val) => CompIn(jointidx, :in)
                    for (i, val) in enumerate(Tuple(assmt))
                )...,
                CompOut(jointidx, :out) => Output(:out => jointidx)
            )
            for (assmt, jointidx) in pairs(LinearIndices(a.size))
        ),
        a
    )