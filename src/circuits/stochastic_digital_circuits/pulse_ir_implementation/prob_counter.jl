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
Circuits.outputs(c::ProbCounter) = NamedValues(
    :count => UnbiasedSpikeCountReal(PulseIR.threshold(c.ti))
)
Circuits.target(::ProbCounter) = Spiking()

Circuits.implement(c::ProbCounter, ::Spiking) =
    CompositeComponent(
        inputs(c), outputs(c),
        (mux=c.mux, ti=c.ti, gate=c.gate),
        (
            Input(:sel) => CompIn(:mux, :sel),
            Input(:samples) => CompIn(:mux, :values),
            CompOut(:gate, :out) => Output(:count),
            (c.output_inverse_prob ? inv_prob_edges(c) : prob_edges(c))...
        )
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

PulseIR.output_windows(c::ProbCounter, d::Dict{Input, Window}) =
    PulseIR.output_windows(implement(c, Spiking()), d)