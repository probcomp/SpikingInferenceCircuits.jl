"""
    ThresholdedIndicator(threshold::Int)

Emits an output spike once it receives its `threshold`th input spike.
"""
struct ThresholdedIndicator <: GenericComponent
    threshold::Int
end
Circuits.target(::ThresholdedIndicator) = Spiking()
Circuits.inputs(::ThresholdedIndicator) = NamedValues(:in => SpikeWire())
Circuits.outputs(::ThresholdedIndicator) = NamedValues(:out => SpikeWire())