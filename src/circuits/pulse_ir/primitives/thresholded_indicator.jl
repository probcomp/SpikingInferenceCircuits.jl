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