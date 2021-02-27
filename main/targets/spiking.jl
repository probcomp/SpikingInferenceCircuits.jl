"""
    Spiking <: Target

Spiking circuit target for information processing.
"""
struct Spiking <: Target end

# TODO: docstrings

struct SpikeWire <: PrimitiveValue{Spiking} end

struct PoissonNeuron <: PrimitiveComponent{Spiking}
    rate::Float64
end
inputs(::PoissonNeuron) = CompositeValue((on=SpikeWire, off=SpikeWire))
outputs(::PoissonNeuron) = CompositeValue((out=SpikeWire,))