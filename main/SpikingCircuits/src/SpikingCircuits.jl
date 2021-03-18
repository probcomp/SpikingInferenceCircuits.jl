module SpikingCircuits
import Circuits

"""
    Spiking <: Target

Spiking circuit target for information processing.
"""
struct Spiking <: Circuits.Target end

# all values in a spiking circuit can be compiled into `SpikeWire`s
Circuits.compiles_to_binary(::Circuits.Value, ::Spiking) = true

### Simulator ###
include("simulator.jl")
const Sim = SpikingSimulator

##############
# Primitives #
##############

struct SpikeWire <: Circuits.PrimitiveValue{Spiking} end
Circuits.abstract(::SpikeWire) = Circuits.Binary()
Circuits.implement(::Circuits.Binary, ::Spiking) = SpikeWire()

using Distributions: Exponential
exponential(rate) = rand(Exponential(1/rate))
include("on_off_poisson_neuron.jl")
include("integrating_poisson.jl")

export Spiking, SpikingSimulator, SpikeWire
export OnOffPoissonNeuron, IntegratingPoisson

end # module