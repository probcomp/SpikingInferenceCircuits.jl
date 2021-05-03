module PulseIR
using Circuits
using SpikingCircuits

const × = *

include("temporal_interface/interface.jl")

include("primitives/primitives.jl")
include("poisson_implementations/poisson_implementations.jl")

end