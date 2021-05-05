struct PulseConditionalSample{S,G,W,T,O} <: ConcretePulseIRPrimitive
    streamsamples::S
    mux_on_gate::G
    wta_offgate::W
    ti::T
    offgate::O
    time_to_first_sample::Float64
    intersample_hold::Float64
    function PulseConditionalSample(s::S, g::G, w::W, t::T, o::O, ttfs::Float64, ih::Float64) where {S,G,W,T,O}
        @assert has_abstract_of_type(s, PulseIR.ConcreteStreamSamples)
        @assert has_abstract_of_type(g, PulseIR.ConcreteAsyncOnGate)
        @assert has_abstract_of_type(w, PulseIR.ConcreteOffGate)
        @assert has_abstract_of_type(t, PulseIR.ConcreteThresholdedIndicator)
        @assert has_abstract_of_type(o, PulseIR.ConcreteOffGate)

        return new{S,G,W,T,O}(s, g, w, t, o, ttfs, ih)
    end
end
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