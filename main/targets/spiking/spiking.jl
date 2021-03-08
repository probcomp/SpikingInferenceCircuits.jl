# TODO: docstrings

##############
# Primitives #
##############

struct SpikeWire <: PrimitiveValue{Spiking} end

struct PoissonNeuron <: PrimitiveComponent{Spiking}
    rate::Float64
end
inputs(::PoissonNeuron) = CompositeValue((on=SpikeWire(), off=SpikeWire()))
outputs(::PoissonNeuron) = CompositeValue((out=SpikeWire(),))

initial_state(::PoissonNeuron) = EmptyState()

### Simulator ###
include("simulator.jl")