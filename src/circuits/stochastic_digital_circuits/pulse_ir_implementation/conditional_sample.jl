struct PulseConditionalSample <: ConcretePulseIRPrimitive
    streamsamples
    wta_offgate
    ti
    offgate
    time_to_first_sample::Float64
    intersample_hold::Float64
    # TODO: constructor; types
end
Circuits.abstract(c::PulseConditionalSample) = ConditionalSample(#=TODO=#)
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
        p.ti,
        p.offgate
    )

Circuits.implement(c::PulseConditionalSample, ::Spiking) =
    CompositeComponent(
        inputs(c), outputs(c),
        (
            ss = c.streamsamples,
            wta = ConcreteWTA(
                output_domain_size(abstract(c)),
                c.wta_offgate
            ),
            counter = probcounter(c)
        ),
        (
            (
                Input(:in_val => i) => CompIn(:ss, i)
                for i=1:in_domain_size(abstract(p))
            )...,
            (
                CompOut(:ss, i) => CompIn(:wta, i)
                for i=1:out_domain_size(abstract(p))
            )...,
            Iterators.flatten(
                (
                    CompOut(:wta, i) => CompIn(:counter, :sel => i),
                    CompOut(:ss, i) => CompIn(:counter, :samples => i)
                )
                for i=1:out_domain_size(abstract(p))
            )...,
            (
                CompOut(:wta, i) => Output(:value => i)
                for i=1:out_domain_size(abstract(p))
            )...,
            CompOut(:counter, :out) => Output(:inverse_prob)
        )
    )

### Temporal Interface ###
PulseIR.failure_probability_bound(::PulseConditionalSample) = error("TODO")
#= Types of error to account for include:
- No samples within p.time_to_first_sample
- Getting a sample within p.intersample_hold
=#

# TODO: I will probably need to manually input a shorter output window for `val`.
function PulseIR.output_windows(p::PulseConditionalSample, d::Dict{Input, Window})
    sample_outs = output_windows(p.streamsamples, Dict{Input, Window}(
        Input(i) => d[Input(:in_val => i)]
        for i=1:out_domain_size(abstract(p))
    ))
    sample_output_window = containing_window(values(sample_outs))
    earliest_sample_output = sample_output_window.interval.min

    first_sample_window = Window(
        Interval(
            searliest_sample_output,
            earliest_sample_output + p.time_to_first_sample
        ),
        sample_output_window.pre_hold,
        p.intersample_hold
    )

    val_outs = output_windows(p.wta, Dict{Input, Window}(
        Input(i) => first_sample_window
        for i=1:out_domain_size(abstract(p))
    ))

    counter_window = output_windows(
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
            Output(:sel => i) => val_outs[Output(i)]
            for i=1:out_domain_size(abstract(p))
        )...
    )
end

PulseIR.valid_strict_inwindows(::PulseConditionalSample, ::Dict{Input, Window}) = error("Not implemented.")

is_valid_input(::PulseConditionalSample, d::Dict{Input, UInt}) = sum(values(d)) == 1