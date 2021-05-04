"""
    OffGate

Repeats spikes in the `:in` wire unless a spike has been received in the `:off` wire recently.
"""
struct OffGate <: GenericComponent end
Circuits.target(::OffGate) = Spiking()
Circuits.inputs(::OffGate) = NamedValues(:in => SpikeWire(), :off => SpikeWire())
Circuits.outputs(::OffGate) = NamedValues(:out => SpikeWire())

"""
    ConcreteOffGate(ΔT, max_delay, M)

`OffGate` with concrete Pulse IR temporal interface.
"""
struct ConcreteOffGate <: ConcretePulseIRPrimitive
    ΔT::Float64
    max_delay::Float64
    M::Float64
end
Circuits.abstract(::ConcreteOffGate) = OffGate()
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::ConcreteOffGate) = Circuits.$s(Circuits.abstract(g)))
end

valid_strict_inwindows(g::ConcreteOffGate, d::Dict{Input, Window}) =
    (
        # OFF window ends before IN window
        d[Input(:in)].interval.min ≥ d[Input(:off)].interval.max &&
        # Total memory of IN and OFF ≤ Neuron memory
        d[Input(:in)].interval.max - d[Input(:off)].interval.min ≤ g.ΔT &&
        # OFF has a hold window through the full output window
        d[Input(:in)].interval.max + g.max_delay ≤ end_of_post_hold(d[Input(:off)]) &&
        # IN has a hold window covering the `maximum_delay` before the output begins
        d[Input(:in)].pre_hold ≥ g.max_delay
    )
output_windows(g::ConcreteOffGate, d::Dict{Input, Window}) =
    Dict(Output(:out) => Window(
        Interval(
            d[Input(:in)].interval.min,
            d[Input(:in)].interval.max + g.max_delay
        ),
        0., 0. # TODO: if we have extra hold time in the inputs, we can probably do better
    ))

# The input is valid so long as there are fewer than M input spikes
is_valid_input(g::ConcreteOffGate, d::Dict{Input, UInt}) = d[Input(:in)] < g.M