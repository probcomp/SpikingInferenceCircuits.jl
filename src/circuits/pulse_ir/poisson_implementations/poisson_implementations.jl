PoissonNeuron(fs, ΔT::Real, λ) =
    SpikingCircuits.InputFunctionPoisson(Tuple(fs), Tuple(ΔT for _ in fs), λ)

include("off_gate.jl")
include("async_on_gate.jl")
include("thresholded_indicator.jl")
include("streamsamples.jl")