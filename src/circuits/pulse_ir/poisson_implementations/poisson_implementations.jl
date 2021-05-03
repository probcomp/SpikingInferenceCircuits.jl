PoissonNeuron(fs, ΔT::Real, λ) =
    SpikingCircuits.InputFunctionPoisson(fs, fill(ΔT, length(fs)), λ)

include("off_gate.jl")
include("async_on_gate.jl")
include("thresholded_indicator.jl")
