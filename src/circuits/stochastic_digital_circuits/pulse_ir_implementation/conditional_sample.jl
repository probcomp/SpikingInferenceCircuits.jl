struct PulseConditionalSample{S,G,W,T,O} <: ConcretePulseIRPrimitive
    streamsamples::S
    mux_on_gate::G
    wta_offgate::W
    ti::T
    offgate::O
    time_to_first_sample::Float64
    intersample_hold::Float64
    function PulseConditionalSample(s::S, g::G, w::W, t::T, o::O, ttfs::Real, ih::Real) where {S,G,W,T,O}
        @assert has_abstract_of_type(s, PulseIR.ConcreteStreamSamples)
        @assert has_abstract_of_type(g, PulseIR.ConcreteAsyncOnGate)
        @assert has_abstract_of_type(w, PulseIR.ConcreteOffGate)
        @assert has_abstract_of_type(t, PulseIR.ConcreteThresholdedIndicator)
        @assert has_abstract_of_type(o, PulseIR.ConcreteOffGate)

        return new{S,G,W,T,O}(s, g, w, t, o, ttfs, ih)
    end
end

### Constructors ###

ConcretePulseConditionalSample(
    P::Matrix{Float64},
    K::Int,
    ss_params::Tuple{Int, Function},
    mux_on_params::Tuple{Int, Real, Int},
    wta_off_params::Tuple{Int, Real, Int},
    ti_params::Tuple{Int, Real, Int},
    off_params::Tuple{Int, Real, Int},
    time_to_first_sample::Real,
    intersample_hold::Real
) = SDCs.PulseConditionalSample(
        PulseIR.ConcreteStreamSamples(P, ss_params...),
        PulseIR.ConcreteAsyncOnGate(mux_on_params...),
        PulseIR.ConcreteOffGate(wta_off_params...),
        PulseIR.ConcreteThresholdedIndicator(K + 1, ti_params...),
        PulseIR.ConcreteOffGate(off_params...),
        time_to_first_sample,
        intersample_hold
    )

ConcretePulseConditionalSample(
    P::Matrix{Float64}, K::Int, ss_samplecount_dist::Function, ΔT::Real, max_delay::Real, M::Int,
    ttfs::Real, ih::Real
) = ConcretePulseConditionalSample(
        P, K, (ΔT, ss_samplecount_dist), (ΔT, max_delay, M), (ΔT, max_delay, M),
        (ΔT, max_delay, M), (ΔT, max_delay, M), ttfs, ih
    )
ConcretePulseConditionalSample(
    P::Matrix{Float64}, K::Int, ss_samplecount_rate::Real, args...
) = ConcretePulseConditionalSample(
        P, K, (T -> Distributions.Poisson(T * ss_samplecount_rate)), args...
    )
ConcretePulseConditionalSample(cs::ConditionalSample, args...) =
    ConcretePulseConditionalSample(cs.P, args...)

PoissonPulseConditionalSample(
    concrete_pcs::PulseConditionalSample,
    ss_off_rate::Real,
    mux_on_R::Real, wta_off_R::Real,
    ti_R::Real, off_R::Real
) = SDCs.PulseConditionalSample(
        PulseIR.PoissonStreamSamples(concrete_pcs.streamsamples, ss_off_rate),
        PulseIR.PoissonAsyncOnGate(concrete_pcs.mux_on_gate, mux_on_R),
        PulseIR.PoissonOffGate(concrete_pcs.wta_offgate, wta_off_R),
        PulseIR.PoissonThresholdedIndicator(concrete_pcs.ti, ti_R),
        PulseIR.PoissonOffGate(concrete_pcs.offgate, off_R),
        concrete_pcs.time_to_first_sample,
        concrete_pcs.intersample_hold
    )

# Constructor to set all `R` values to be the same, except for the TI's,
# which will be larger to ensure the gate is turned off before
# the spike which turned it off can pass.
PoissonPulseConditionalSample(
    concrete_pcs::PulseConditionalSample,
    off_rate::Real,
    non_ti_R::Real
) = PoissonPulseConditionalSample(
        concrete_pcs, off_rate, non_ti_R, non_ti_R, non_ti_R + 10, non_ti_R
    )

PoissonPulseConditionalSample(concrete_pcs_args::Tuple, args...) =
    PoissonPulseConditionalSample(ConcretePulseConditionalSample(concrete_pcs_args...), args...)

### Circuit interface ###

Circuits.abstract(c::PulseConditionalSample) = ConditionalSample(abstract_to_type(c.streamsamples, PulseIR.StreamSamples).P)
Circuits.inputs(c::PulseConditionalSample) =
    implement_deep(inputs(abstract(c)), Spiking())
Circuits.outputs(c::PulseConditionalSample) =
    NamedValues(
        :value => IndexedValues(SpikeWire() for _=1:out_domain_size(abstract(c))),
        :inverse_prob =>  outputs(probcounter(c))[:count]
    )

probcounter(c::PulseConditionalSample) = ProbCounter(
        true,
        PulseMux(
            Mux(out_domain_size(abstract(c)), SpikeWire()),
            c.mux_on_gate
        ),
        c.ti,
        c.offgate
    )
wta(c::PulseConditionalSample) = PulseIR.ConcreteWTA(
        out_domain_size(abstract(c)),
        c.wta_offgate
    )

Circuits.implement(c::PulseConditionalSample, ::Spiking) =
    CompositeComponent(
        inputs(c), outputs(c),
        (
            ss = c.streamsamples,
            wta = wta(c),
            counter = probcounter(c)
        ),
        (
            (
                Input(:in_val => i) => CompIn(:ss, i)
                for i=1:in_domain_size(abstract(c))
            )...,
            (
                CompOut(:ss, i) => CompIn(:wta, i)
                for i=1:out_domain_size(abstract(c))
            )...,
            Iterators.flatten(
                (
                    CompOut(:wta, i) => CompIn(:counter, :sel => i),
                    CompOut(:ss, i) => CompIn(:counter, :samples => i)
                )
                for i=1:out_domain_size(abstract(c))
            )...,
            (
                CompOut(:wta, i) => Output(:value => i)
                for i=1:out_domain_size(abstract(c))
            )...,
            CompOut(:counter, :count) => Output(:inverse_prob)
        ),
        c
    )

### Temporal Interface ###
PulseIR.failure_probability_bound(::PulseConditionalSample) = error("TODO")
#= Types of error to account for include:
- No samples within p.time_to_first_sample
- Getting a sample within p.intersample_hold
=#

function PulseIR.output_windows(p::PulseConditionalSample, d::Dict{Input, Window})
    sample_outs = PulseIR.output_windows(p.streamsamples, Dict{Input, Window}(
        Input(i) => d[Input(:in_val => i)]
        for i=1:out_domain_size(abstract(p))
    ))
    
    # TODO: If there can be delay on the sample output windows, we will need to account for it
    inwindow = PulseIR.containing_window(values(d))
    first_sample_window = Window(
        Interval(
            inwindow.interval.min,
            inwindow.interval.max + p.time_to_first_sample
        ),
        inwindow.pre_hold,
        p.intersample_hold
    )

    val_outs = PulseIR.output_windows(wta(p), Dict{Input, Window}(
        Input(i) => first_sample_window
        for i=1:out_domain_size(abstract(p))
    ))

    counter_window = PulseIR.output_windows(
        probcounter(p),
        Dict{Input, Window}(Iterators.flatten(
            (
                Input(:sel => i) => val_outs[Output(i)],
                Input(:samples => i) => sample_outs[Output(i)]
            )
            for i=1:out_domain_size(abstract(p))
        ))
    )[Output(:count)]

    return Dict{Output, Window}(
        Output(:inverse_prob) => counter_window,
        (
            Output(:value => i) => val_outs[Output(i)]
            for i=1:out_domain_size(abstract(p))
        )...
    )
end

PulseIR.valid_strict_inwindows(::PulseConditionalSample, ::Dict{Input, Window}) = error("Not implemented.")

is_valid_input(::PulseConditionalSample, d::Dict{Input, UInt}) = sum(values(d)) == 1