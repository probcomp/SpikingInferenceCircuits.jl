"""
    ThresholdedIndicator(threshold::Int)

Emits an output spike once it receives its `threshold`th input spike in the input window.

In a non-failing response to a valid input, should never emit more than 1 output.
"""
# Possibilities for changing the interface (/introducing other components with similar but different interfaces):
# It would be easy enough to have it immediately reset after spiking,
# so it spikes every time it gets a new set of `threshold` spikes.
# Would also be easy to have it spike after the first `K`, then spike again
# every additional `T`.  (Ie. the first threshold and subsequent threshoulds could be different.)
struct ThresholdedIndicator <: GenericComponent
    threshold::Int
end
Circuits.target(::ThresholdedIndicator) = Spiking()
Circuits.inputs(::ThresholdedIndicator) = NamedValues(:in => SpikeWire())
Circuits.outputs(::ThresholdedIndicator) = NamedValues(:out => SpikeWire())

struct ConcreteThresholdedIndicator <: ConcretePulseIRPrimitive
    threshold::Int
    ΔT::Float64 # Neuron memory
    M::Float64 # Number of spikes needed to override & produce another spike after first spike emitted
    max_delay::Float64
end
Circuits.abstract(t::ConcreteThresholdedIndicator) = ThresholdedIndicator(t.threshold)
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(t::PoissonThresholdedIndicator) = Circuits.$s(Circuits.abstract(t)))
end

valid_strict_inwindows(t::ConcreteThresholdedIndicator, d::Dict{Input, Window}) =
    (
        d[Input(:in)].pre_hold ≥ t.ΔT &&
        interval_length(d[Input(:in)]) ≤ t.ΔT - t.max_delay
    )
output_windows(t::ConcreteThresholdedIndicator, d::Dict{Input, Window}) =
    Dict(Output(:out) => Window(
        Interval(
            d[Input(:in)].interval.min,
            d[Inpt(:in)].interval.max + t.max_delay
        ),
        0., 0. # TODO: if we have extra hold time in the inputs, we can probably do better
    ))

# The input is valid unless after passing the threshold, it was fed enough spikes
# that it could override the indicator being OFF.
is_valid_input(g::ConcreteThresholdedIndicator, d::Dict{Input, UInt}) =
    d[Input(:in)] < g.threshold + g.M