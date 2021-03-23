struct SpikingToAssmts{n} <: GenericComponent
    size::NTuple{n, Int}
end
Circuits.abstract(s::SpikingToAssmts) = ToAssmts(s.size)
Circuits.implement(a::ToAssmts, ::Spiking) = SpikingToAssmts(a.size)
Circuits.target(::SpikingToAssmts) = Spiking()
Circuits.inputs(s::SpikingToAssmts) = implement(inputs(abstract(s)), Spiking())
Circuits.outputs(s::SpikingToAssmts) = implement(outputs(abstract(s)), Spiking())
num_inputs(::SpikingToAssmts{n}) where {n} = n

Circuits.implement(s::SpikingToAssmts, ::Spiking) =
    let output = implement_deep(outputs(s), Spiking())
        CompositeComponent(
            implement_deep(inputs(s), Spiking()), output,
            # ThresholdedCounter spikes once it receives `n` input spikes; this acts as an AND
            # gate signifying that all inputs have spiked
            Tuple(ThresholdedCounter(num_inputs(s)) for _=1:length(output[:out])),
            Iterators.flatten(
                Iterators.flatten((
                    (
                        Input(i => val) => CompIn(jointidx, :in)
                        for (i, val) in enumerate(Tuple(assmt))
                    ),
                    (CompOut(jointidx, :out) => Output(:out => jointidx),)
                ))
                for (assmt, jointidx) in pairs(LinearIndices(s.size))
            ),
        )
                end