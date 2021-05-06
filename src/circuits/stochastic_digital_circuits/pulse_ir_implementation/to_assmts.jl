struct PulseToAssmts{n} <: ConcretePulseIRPrimitive
    size::NTuple{n, Int}
    ti_params::Tuple
end
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

Circuits.implement(a::PulseToAssmts) =
    CompositeComponent(
        inputs(a), outputs(a),
        Tuple(
            ThresholdedIndicator(
                length(a.size), a.ti_params...
            ) for _=1:length(outputs(a)[:out])
        ),
        Iterators.flatten(
            (
                (
                    Input(i => val) => CompIn(jointidx, :in)
                    for (i, val) in enumerate(tuple(assmt))
                )...,
                CompOut(jointidx, :out) => Output(:out => jointidx)
            )
            for (assmt, jointidx) in pairs(LinearIndices(a.size))
        ),
        a
    )

### Pulse IR
PulseIR.output_windows(a::PulseToAssmts, d::Dict{Input, Window}) =
    PulseIR.output_windows(implement(a, Spiking()), d)