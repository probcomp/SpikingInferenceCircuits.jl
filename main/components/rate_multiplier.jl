struct RateMultiplier <: GenericComponent
    memory::Float64
    reference_rate::Float64
    inputs::Tuple{Vararg{SpikeRateReal}}
end
RateMultiplier(memory::Float64, inputs::Tuple{Vararg{SpikeRateReal}}) =
    RateMultiplier(memory, prod(r.reference_rate for r in inputs), inputs)

Circuits.target(::RateMultiplier) = Spiking()
Circuits.inputs(m::RateMultiplier) = CompositeValue(m.inputs)

# TODO: maybe have this component not have reference rates at all,
# and have a different component called a `SpikeRateRealMultiplier` which is implemented
# in terms of this, and uses `SpikeRateReal` inputs/outputs?
Circuits.outputs(m::RateMultiplier) = NamedValues(:out => SpikeRateReal(m.reference_rate))

get_neuron(m) =
    let rate_ratio = m.reference_rate / prod(r.reference_rate for r in m.inputs)
        InputFunctionPoisson(
            Tuple(x -> log(x/m.memory) for _ in m.inputs),
            Tuple(m.memory for _ in m.inputs),
            x -> exp(x) * rate_ratio
        )
    end

Circuits.implement(m::RateMultiplier, ::Spiking) =
    CompositeComponent(
        IndexedValues(SpikeWire() for _ in m.inputs),
        NamedValues(:out => SpikeWire()),
        (neuron=get_neuron(m),),
        Iterators.flatten((
            (
                Input(i) => CompIn(:neuron, i)
                for i=1:length(m.inputs)
            ),
            (CompOut(:neuron, :out) => Output(:out),)
        ))
    )
