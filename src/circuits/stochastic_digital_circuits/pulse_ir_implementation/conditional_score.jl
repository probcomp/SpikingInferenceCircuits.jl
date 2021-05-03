"""
Pulse IR implementation of a `ConditionalScore` with a concrete temporal interface.
"""
struct PulseConditionalScore <: ConcretePulseIRPrimitive
    P::Matrix{Float64}
    streamsamples::ConcretePulseIRPrimitive
    mux::ConcretePulseIRPrimitive
    ti::ConcretePulseIRPrimitive
    offgate::ConcretePulseIRPrimitive
end

Circuits.abstract(p::PulseConditionalScore) = ConditionalScore(p.P)

Circuits.inputs(c::PulseConditionalScore) =
    implement_deep(inputs(abstract(c)), Spiking())
Circuits.outputs(c::PulseConditionalScore) =
    implement_deep(outputs(abstract(c)), Spiking())

Circuits.implement(p::PulseConditionalScore) =
    CompositeComponent(
        inputs(p), outputs(p),
        (
            ss=StreamSamples(p.P),
            mux=MUX(out_domain_size(abstract(p)), SpikeWire()),
            ti=ThresholdedIndicator(),
            gate=OffGate()
        ),
        (
            (
                Input(:in_val => i) => CompIn(:ss, i)
                for i=1:in_domain_size(abstract(p))
            )...,
            (
                CompOut(:ss, i) => CompIn(:mux, :in => i)
                for i=1:out_domain_size(abstract(p))
            )...,
            (
                Input(:obs => i) => CompIn(:mux, :sel => i)
                for i=1:out_domain_size(abstract(p))
            )...,
            (
                CompOut(:ss, i) => CompIn(:ti, :in)
                for i=1:out_domain_size(abstract(p))
            )...,
            CompOut(:ti, :out) => CompIn(:gate, :off),
            CompOut(:mux, :out) => CompIn(:gate, :in),
            CompOut(:gate, :out) => Output(:prob)
        ),
        p
    )

### Temporal Interface ###
failure_probability_bound(p::PulseConditionalScore) =
    1 - (1 - p_subcomponent_fails(p))*(1 - p_insufficient_hold_for_gate(p))
p_subcomponent_fails(p::PulseConditionalScore) = 1 - pnf(p.mux)*pnf(p.offgate)*pnf(p.streamsamples)*pnf(p.ti)
pnf(c) #= probability of not failing =# = 1 - failure_probability_bound(c)
p_insufficient_hold_for_gate(p::PulseConditionalScore) = error("TODO")

output_windows(p::PulseConditionalScore, d::Dict{Input, Window}) = output_windows(implement(p, Spiking()), d)
valid_strict_inwindows(::PulseConditionalScore, ::Dict{Input, Window}) = error("Not implemented.")