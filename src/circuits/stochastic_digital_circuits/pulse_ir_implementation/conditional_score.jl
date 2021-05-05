"""
Pulse IR implementation of a `ConditionalScore` with a concrete temporal interface.
"""
struct PulseConditionalScore{SS, MOG, TI, OG} <: ConcretePulseIRPrimitive
    streamsamples::SS
    mux_on_gate::MOG
    ti::TI
    offgate::OG
    function PulseConditionalScore(s::S,m::M,t::T,o::O) where {S,M,T,O}
        @assert has_abstract_of_type(s, PulseIR.ConcreteStreamSamples)
        @assert has_abstract_of_type(m, PulseIR.ConcreteAsyncOnGate)
        @assert has_abstract_of_type(t, PulseIR.ConcreteThresholdedIndicator)
        @assert has_abstract_of_type(o, PulseIR.ConcreteOffGate)

        return new{S,M,T,O}(s,m,t,o)
    end
end

Circuits.abstract(p::PulseConditionalScore) = ConditionalScore(p.streamsamples.P)

Circuits.inputs(c::PulseConditionalScore) =
    implement_deep(inputs(abstract(c)), Spiking())
Circuits.outputs(c::PulseConditionalScore) =
    NamedValues(:prob => outputs(probcounter(c))[:count])

probcounter(p::PulseConditionalScore) =
    ProbCounter(
        false,
        PulseMux(
            Mux(out_domain_size(abstract(p)), SpikeWire()),
            p.mux_on_gate
        ),
        p.ti,
        p.offgate
    )

Circuits.implement(p::PulseConditionalScore, ::Spiking) =
    CompositeComponent(
        inputs(p), outputs(p),
        (
            ss=p.streamsamples,
            counter=probcounter(p)
        ),
        (
            (
                Input(:in_val => i) => CompIn(:ss, i)
                for i=1:in_domain_size(abstract(p))
            )...,
            (
                CompOut(:ss, i) => CompIn(:counter, :samples => i)
                for i=1:out_domain_size(abstract(p))
            )...,
            (
                Input(:obs => i) => CompIn(:counter, :sel => i)
                for i=1:out_domain_size(abstract(p))
            )...,
            CompOut(:counter, :count) => Output(:prob)
        ),
        p
    )

### Temporal Interface ###
# TODO: for failure probability, also factor in possibility that we haven't gotten K
# spikes by the end of ΔT
# PulseIR.failure_probability_bound(p::PulseConditionalScore) =
    # 1 - (1 - p_subcomponent_fails(p))*(1 - p_insufficient_hold_for_gate(p))
# p_subcomponent_fails(p::PulseConditionalScore) = 1 - pnf(p.mux)*pnf(p.offgate)*pnf(p.streamsamples)*pnf(p.ti)
# pnf(c) #= probability of not failing =# = 1 - PulseIR.failure_probability_bound(c)
# p_insufficient_hold_for_gate(p::PulseConditionalScore) = error("TODO")

PulseIR.output_windows(p::PulseConditionalScore, d::Dict{Input, Window}) = PulseIR.output_windows(implement(p, Spiking()), d)
PulseIR.valid_strict_inwindows(::PulseConditionalScore, ::Dict{Input, Window}) = error("Not implemented.")