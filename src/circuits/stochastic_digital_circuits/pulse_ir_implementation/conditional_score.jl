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

### Constructors ###
 
ConcretePulseConditionalScore(
    P::Matrix{Float64},
    K::Int,
    ss_params::Tuple{Int, Function},
    mux_on_params::Tuple{Int, Real, Int},
    ti_params::Tuple{Int, Real, Int},
    off_params::Tuple{Int, Real, Int},
) = SDCs.PulseConditionalScore(
        PulseIR.ConcreteStreamSamples(P, ss_params...),
        PulseIR.ConcreteAsyncOnGate(mux_on_params...),
        PulseIR.ConcreteThresholdedIndicator(K, ti_params...),
        PulseIR.ConcreteOffGate(off_params...)
    )
ConcretePulseConditionalScore(
    P::Matrix{Float64}, K::Int, ss_samplecount_dist::Function, ΔT::Real, max_delay::Real, M::Int
) = ConcretePulseConditionalScore(
        P, K, (ΔT, ss_samplecount_dist), (ΔT, max_delay, M), (ΔT, max_delay, M), (ΔT, max_delay, M)
    )
    ConcretePulseConditionalScore(
    P::Matrix{Float64}, K::Int, ss_samplecount_rate::Real, args...
) = ConcretePulseConditionalScore(
        P, K, (T -> Distributions.Poisson(T * ss_samplecount_rate)), args...
    )
ConcretePulseConditionalScore(cs::ConditionalScore, args...) =
    ConcretePulseConditionalScore(cs.P, args...)  

PoissonPulseConditionalScore(
    concrete_pcs::PulseConditionalScore,
    ss_off_rate::Real, mux_on_R::Real, ti_R::Real, off_R::Real
) = SDCs.PulseConditionalScore(
        PulseIR.PoissonStreamSamples(concrete_pcs.streamsamples, ss_off_rate),
        PulseIR.PoissonAsyncOnGate(concrete_pcs.mux_on_gate, mux_on_R),
        PulseIR.PoissonThresholdedIndicator(concrete_pcs.ti, ti_R),
        PulseIR.PoissonOffGate(concrete_pcs.offgate, off_R)
    )

# Constructor to set the TI's `R` value to the given quantity,
# and set the others to be larger.
# This will ensure that the gate outputs spikes before
# it is turned off.
PoissonPulseConditionalScore(
    concrete_pcs::PulseConditionalScore,
    off_rate::Real,
    ti_R::Real
) = PoissonPulseConditionalScore(
        concrete_pcs, off_rate, ti_R + 10, ti_R, ti_R + 10
    )
PoissonPulseConditionalScore(concrete_pcs_args::Tuple, args...) =
    PoissonPulseConditionalScore(ConcretePulseConditionalScore(concrete_pcs_args...), args...)

### Circuits methods ###

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

#=
Temporal Interface automatically inferred from implementation is correct.

However, we will have to manually override `failure_probability_bound` for this component.
=#