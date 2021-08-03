struct PoissonAutoNormalizedMultiply{TI, GATE} <: ConcretePulseIRPrimitive
    input_line_denominators::Vector{Vector{Float64}}
    base::Float64
    memory::Float64
    renormalization_rate::Float64
    ti::TI
    gate::GATE
end
PoissonAutoNormalizedMultiply(input_line_denominators, base, k, memory, renormalization_rate, ti_params, gate_params) =
    PoissonAutoNormalizedMultiply(
        convert(Vector{Vector{Float64}}, input_line_denominators),
        map(x -> convert(Float64, x), (base, memory, renormalization_rate))...,
        PoissonThresholdedIndicator(k, ti_params...),
        PoissonOffGate(gate_params...)
    )
Circuits.inputs(m::PoissonAutoNormalizedMultiply) = NamedValues(
    :ind => SpikeWire(), # indicate that all values have come in
    :counts => IndexedValues(
        IndexedValues(
            SpikeWire() for _ in denoms
        ) for denoms in m.input_line_denominators
    )
)
Circuits.outputs(m::PoissonAutoNormalizedMultiply) = NamedValues(
    :scale => SpikeWire(), # spike count s.t. value_i = `scaled_rates[i] * base^(-scale)`
    :ind => SpikeWire(), # spikes once `scale` count has been emitted & `scaled_rates` are stable
    :scaled_rates => IndexedValues(SpikeWire() for _ in m.input_line_denominators)
)

Circuits.implement(m::PoissonAutoNormalizedMultiply, ::Spiking) =
    CompositeComponent(
        inputs(m), outputs(m),
        (
            multipliers=IndexedComponentGroup(
                PoissonNeuron(
                    [
                        (c -> log(c / denominator) for denominator in denoms)...,
                        c -> log(m.base) * c
                    ],
                    m.memory,
                    exp # TODO: do I want to have a ``denominator'' for the output?
                )
                for denoms in m.input_line_denominators
            ),
            ti=m.ti, gate=m.gate, and=async_on_gate(m.gate),
            renormalization_spiker=PoissonNeuron(
                [c -> c], m.memory, u -> u > 0 ? m.renormalization_rate : 0.
            )
        ), (
            ( # Input counts --> Multipliers
                Input(:counts => clusteridx => factoridx) => CompIn(:multipliers => clusteridx, factoridx)
                for (clusteridx, denoms) in enumerate(m.input_line_denominators)
                    for factoridx = 1:length(denoms)
            )...,
            # Input ind turns on renormalization_spiker
            Input(:ind) => CompIn(:renormalization_spiker, 1),
            
            ( # Multiplier outputs --> Output rates
                CompOut(:multipliers => clusteridx, :out) => Output(:scaled_rates => clusteridx)
                for clusteridx = 1:length(m.input_line_denominators)
            )...,

            ( # Multiplier outputs --> TI input
                CompOut(:multipliers => clusteridx, :out) => CompIn(:ti, :in)
                for clusteridx=1:length(m.input_line_denominators)
            )...,

            CompOut(:renormalization_spiker, :out) => CompIn(:gate, :in),
            CompOut(:ti, :out) => CompIn(:gate, :off),

            CompOut(:ti, :out) => CompIn(:and, :in),
            Input(:ind) => CompIn(:and, :on),
            CompOut(:and, :out) => Output(:ind),

            ( # renormalization --> multipliers
                CompOut(:gate, :out) => CompIn(:multipliers => clusteridx, length(denoms) + 1)
                for (clusteridx, denoms) in enumerate(m.input_line_denominators)
            )...,
            CompOut(:gate, :out) => Output(:scale)
        )
    )