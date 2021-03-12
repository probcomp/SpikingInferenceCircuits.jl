module SpikingCircuits
import Circuits

"""
    Spiking <: Target

Spiking circuit target for information processing.
"""
struct Spiking <: Circuits.Target end

### Simulator ###
include("simulator.jl")
const Sim = SpikingSimulator

##############
# Primitives #
##############

struct SpikeWire <: Circuits.PrimitiveValue{Spiking} end

include("poisson_neuron.jl")

export Spiking, SpikingSimulator, SpikeWire
export PoissonNeuron

end # module