module SDCs
using Circuits
using SpikingCircuits
using ..PulseIR
using ..PulseIR: ConcretePulseIRPrimitive

# TODO: this file should probably be split up or live somewhere different
include("../value_types.jl")

include("primitives/mux.jl")
include("primitives/conditional_score.jl")
include("primitives/conditional_sample.jl")

include("pulse_ir_implementation/mux.jl")
include("pulse_ir_implementation/prob_counter.jl")
include("pulse_ir_implementation/conditional_score.jl")
include("pulse_ir_implementation/conditional_sample.jl")

end