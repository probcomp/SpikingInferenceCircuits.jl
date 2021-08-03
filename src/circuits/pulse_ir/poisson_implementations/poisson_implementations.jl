PoissonNeuron(fs, ΔT::Real, λ) =
    SpikingCircuits.InputFunctionPoisson(Tuple(fs), Tuple(ΔT for _ in fs), λ)

# linear between minx and maxx; stays at maxy above maxx and at miny below minx
truncated_linear(miny, maxy, minx, maxx) =
    x -> x ≥ maxx ? maxy :
         x ≤ minx ? miny :
                    (x - minx) * (maxy - miny)/(maxx - minx) + miny

include("off_gate.jl")
include("async_on_gate.jl")
include("thresholded_indicator.jl")
include("streamsamples.jl")
include("theta.jl")

include("timer.jl") # Util

include("multiplier.jl")
include("sync.jl")

include("auto_normalized_multiply.jl")