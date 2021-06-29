module PulseIR
using Base: Float64
using Circuits
using SpikingCircuits
import Distributions

const × = *

include("temporal_interface/interface.jl")

include("primitives/primitives.jl")
include("poisson_implementations/poisson_implementations.jl")

export Interval, Window, ConcretePulseIRPrimitive

end