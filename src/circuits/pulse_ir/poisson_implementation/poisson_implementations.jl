PoissonNeuron(fs, ΔT::Real, λ) =
    SpikingCircuits.InputFunctionPoisson(fs, fill(ΔT, length(fs)), λ)

include("gated_repeater.jl")