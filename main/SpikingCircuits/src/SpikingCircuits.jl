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
Circuits.abstract(::SpikeWire) = Circuits.Binary()
Circuits.implement(::Circuits.Binary, ::Spiking) = SpikeWire()

include("on_off_poisson_neuron.jl")
include("integrating_poisson.jl")

export Spiking, SpikingSimulator, SpikeWire
export OnOffPoissonNeuron, IntegratingPoisson

end # module