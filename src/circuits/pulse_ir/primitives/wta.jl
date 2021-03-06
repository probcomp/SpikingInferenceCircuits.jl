# Question: should this be in `pulse_ir/primitives/` or
# in `stochastic_digital_circuits/pulse_ir_implementation/`?

"""
    WTA(n)

Winner takes all circuit with `n` input values.  Repeats the first spike passed in
(so long as there is sufficient time before the second spike).

Currently this is a Combinatory component which expects to receive 1 input spike,
repeats that spike, and then has a long output hold window.
There may be better ways to understand this component which I haven't figured out yet.
"""
struct WTA <: GenericComponent
    n_inputs::Int
end
Circuits.target(::WTA) = Spiking()
Circuits.inputs(w::WTA) = IndexedValues(SpikeWire() for _=1:w.n_inputs)
Circuits.outputs(w::WTA) = Circuits.inputs(w)
# TODO: `fail` wire output for when it fails to distinguish the first input

struct ConcreteWTA{G} <: ConcretePulseIRPrimitive
    n_inputs::Int
    offgate::G
    function ConcreteWTA(n_inputs::Int, off::G) where {G}
        @assert has_abstract_of_type(off, ConcreteOffGate)
        return new{G}(n_inputs, off)
    end
end
Circuits.abstract(w::ConcreteWTA) = WTA(w.n_inputs)
Circuits.target(w::ConcreteWTA) = target(abstract(w))
Circuits.inputs(w::ConcreteWTA) = inputs(abstract(w))
Circuits.outputs(w::ConcreteWTA) = outputs(abstract(w))

Circuits.implement(w::ConcreteWTA, ::Spiking) =
    CompositeComponent(
        inputs(w), outputs(w),
        (
            offs=IndexedComponentGroup(w.offgate for _=1:w.n_inputs),
            on=async_on_gate(w.offgate) # construct an On gate with same params as the Off gate
            # TODO: support different On gate params.  Also TODO: better constructors!
        ),
        Iterators.flatten((
            (
                Input(i) => CompIn(:offs => i, :in)
                for i=1:w.n_inputs
            ),
            ( # Input spike immediately turns of all other repeaters
                Input(i) => CompIn(:offs => j, :off)
                for i=1:w.n_inputs
                    for j=1:w.n_inputs if j != i
            ),
            ( # After spike is repeated, that repeater is turned off
                CompOut(:offs => i, :out) => CompIn(:offs => i, :off)
                for i=1:w.n_inputs
            ),
            (
                CompOut(:offs => i, :out) => Output(i)
                for i=1:w.n_inputs
            ),

            # Ongate is activated by any output spike from the WTA,
            # and passes any input spike to become an "OFF" spike for each offgate
            # This ensures that the OFF gates are turned off by every IN spike
            # that occurs within ongate.??T of the WTA's output,
            # and thus only one spike will be emitted so long as input spikes stop
            # within ongate.??T of the wta's output.
            (
                Input(i) => CompIn(:on, :in)
                for i=1:w.n_inputs
            ),
            (
                CompOut(:offs => i, :out) => CompIn(:on, :on)
                for i=1:w.n_inputs
            ),
            (
                CompOut(:on, :out) => CompIn(:offs => i, :off)
                for i=1:w.n_inputs
            )
        )),
        w
    )

### Temporal Interface ###
failure_probabability_bound(w::ConcreteWTA) = failure_probabability_bound(implement(w, Spiking()))

valid_strict_inwindows(w::ConcreteWTA, d::Dict{Input, Window}) =
    let inwindow = containing_window(d[Input(i)] for i=1:length(inputs(w)))
        error("Not implemented.")
        # inwindow.pre_hold ??? off_memory(c.offgate) && 
        # in_window.post_hold ??? [time until we are certain a spike has been received]
        # With the current no-delay implementation, we can be certain a spike turns off the other neurons
        # immediately.  In the future we will need to account for this!
        # TODO
    end

function output_windows(w::ConcreteWTA, d::Dict{Input, Window})
    inwindow = containing_window(d[Input(i)] for i=1:length(inputs(w)))
    conc_off = abstract_to_type(w.offgate, ConcreteOffGate)
    out_interval = Interval( # outputs 
        inwindow.interval.min,
        inwindow.interval.max + conc_off.max_delay
    )
    outwindow = Window(
        out_interval, Inf,
        max(0, conc_off.??T - interval_length(out_interval))
    )
    return Dict{Output, Window}(
        Output(i) => outwindow
        for i=1:length(outputs(w))
    )
end

# An input to a WTA is only considered valid if there is exactly 1 input.
is_valid_input(::ConcreteWTA, d::Dict{Input, UInt}) = sum(values(d)) == 1

#=
One thing this is not accounting for is that it is possible for enough input spikes
to be sent into an `OffGate` to override the off signal.
Thus, post-output hold time is not absolute.

In general I don't love the implementation & conceptual picture of this component here.
=#