struct PulseConditionalScore <: GenericComponent
    P::Matrix{Float64}
    streamsamples
    mux
    ti
    offgate
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
p_subcomponent_fails(p::PulseConditionalScore) = error("TODO")
p_insufficient_hold_for_gate(p::PulseConditionalScore) = error("TODO")

# Shouldn't need these at all:
# ss_outwindow(p::PulseConditionalScore, d::Dict{Input, Window}) =
#     (output_windows(p.streamsamples, Dict(Input(:in) => d[Input(:in_val)]))
#         |> values |> containing_window)

# mux_outwindow(p::PulseConditionalScore, d::Dict{Input, Window}) =
#     output_windows(p.mux, Dict(:))

# ti_outwindow(p::PusleConditionalScore, d::Dict{Input, Window}) =
#     output_windows(
#         p.ti, Dict(Input(:in) => mux_outwindow(p, d))
#     )[Output(:out)]

# initial_off_gate_input(p, d) = Dict(
#     Input(:off) => Window(
#         Interval(ti_outwindow(p, d).interval.min, ti_outwindow(p, d).interval.min)
#         Inf, Inf
#     ),
#     Input(:in) => ss_outwindow(p, d)
# )
# can_support_inwindows(p::PulseConditionalScore, d::Dict{Input, Window}) = (
#     can_support_inwindows(p.streamsamples, Dict(Input(:in) => d[Input(:in_val)])) &&
#     can_support_inwindows(p.mux, Dict(
#         (Input(:in => i) => ss_outwindow(p, d) for i=1:out_domain_size(abstract(p)))...,
#         (Input(:sel => i) => d[Input(:obs => i)] for i=1:out_domain_size(abstract(p)))...
#     ) &&
#     # make sure the gate could remember an OFF input through the whole interval
#     initial_off_gate_input(p.gate, no_off_gate_input(p, d))
# )

output_windows(p::PulseConditionalScore, d::Dict{Input, Window}) = output_windows(implement(p, Spiking()), d)
valid_strict_inwindows(::PulseConditionalScore, ::Dict{Input, Window}) = error("Not implemented.")