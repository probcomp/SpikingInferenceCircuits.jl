# TODO: docstrings

### Simulator ###
include("simulator.jl")
const SSim = SpikingSimulator

##############
# Primitives #
##############

struct SpikeWire <: PrimitiveValue{Spiking} end

include("poisson_neuron.jl")