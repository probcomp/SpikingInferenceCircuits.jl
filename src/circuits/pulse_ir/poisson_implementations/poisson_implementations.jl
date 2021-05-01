# Implementations of the primitives and other non-primitive Pulse IR components

PoissonNeuron(fs, ΔT::Real, λ) =
    SpikingCircuits.InputFunctionPoisson(fs, fill(ΔT, length(fs)), λ)

# primitives
include("off_gate.jl")
include("async_on_gate.jl")
include("thresholded_indicator.jl")

# non-primitives
include("mux.jl")