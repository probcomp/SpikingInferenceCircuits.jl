module SDCs
using Circuits
using SpikingCircuits
using ..PulseIR
using ..PulseIR: ConcretePulseIRPrimitive
import Distributions

# TODO: do we want the SDCs module to contain Spiking-specific code?
# Such as the Pulse IR implementations and the Spiking-specific value types?

include("value_types.jl")

include("primitives/mux.jl")
include("primitives/conditional_score.jl")
include("primitives/conditional_sample.jl")
include("primitives/to_assmts.jl")
include("primitives/lookup_table.jl")

include("pulse_ir_implementation/mux.jl")
include("pulse_ir_implementation/prob_counter.jl")
include("pulse_ir_implementation/conditional_score.jl")
include("pulse_ir_implementation/conditional_sample.jl")
include("pulse_ir_implementation/to_assmts.jl")

include("nonprimitives/cpt_sample_score.jl")

export FiniteDomainValue, PositiveReal

end