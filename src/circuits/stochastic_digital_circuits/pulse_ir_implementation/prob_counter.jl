struct ProbCounter <: ConcretePulseIRPrimitive
    output_inverse_prob::Bool
    mux
    ti
    gate
end
Circuits.inputs(c::ProbCounter) = NamedValues(
    :samples => inputs(c.mux)[:values],
    :sel => inputs(c.mux)[:sel]
)
Circuits.outputs(c::ProbCounter) =
    let K = PulseIR.threshold(c.ti)
        NamedValues(
            :count => IndicatedSpikeCountReal(
                UnbiasedSpikeCountReal(c.output_inverse_prob ? K - 1 : K)
            )
        )
    end
Circuits.target(::ProbCounter) = Spiking()

# TODO: move the logic about K vs K + 1, and the relative rates of the TI vs the Mux,
# to this file!
# Can we have more convenient constructors which let us handle that all automatically?
# Update -- I think this is mostly done?  But I should document it better.

Circuits.implement(c::ProbCounter, ::Spiking) =
    CompositeComponent(
        inputs(c), implement_deep(outputs(c), Spiking()),
        (mux=c.mux, ti=c.ti, gate=c.gate),
        (
            Input(:sel) => CompIn(:mux, :sel),
            Input(:samples) => CompIn(:mux, :values),
            CompOut(:gate, :out) => Output(:count => :count),
            CompOut(:ti, :out) => Output(:count => :ind), # indicate when the count is done being emitted
            (c.output_inverse_prob ? inv_prob_edges(c) : prob_edges(c))...
        ), c
    )
inv_prob_edges(c) = (
    (
        Input(:samples => i) => CompIn(:gate, :in)
        for i=1:length(inputs(c)[:samples])
    )...,
    CompOut(:mux, :out) => CompIn(:ti, :in),
    CompOut(:ti, :out) => CompIn(:gate, :off)
)
prob_edges(c) = (
    (
        Input(:samples => i) => CompIn(:ti, :in)
        for i=1:length(inputs(c)[:samples]) 
    )...,
    CompOut(:mux, :out) => CompIn(:gate, :in),
    CompOut(:ti, :out) => CompIn(:gate, :off)
)