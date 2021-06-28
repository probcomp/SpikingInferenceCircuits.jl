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
include("primitives/multiplication.jl")
include("primitives/theta.jl")
include("primitives/step.jl")
include("primitives/passer_blocker.jl")

include("pulse_ir_implementation/mux.jl")
include("pulse_ir_implementation/prob_counter.jl")
include("pulse_ir_implementation/conditional_score.jl")
include("pulse_ir_implementation/conditional_sample.jl")
include("pulse_ir_implementation/to_assmts.jl")
include("pulse_ir_implementation/multiplication.jl")
include("pulse_ir_implementation/theta.jl")
include("pulse_ir_implementation/step.jl")
include("pulse_ir_implementation/passer_blocker.jl")

include("nonprimitives/cpt_sample_score.jl")
include("nonprimitives/multi_input_lookup_table.jl")

export FiniteDomainValue, NonnegativeReal, SingleNonnegativeReal, ProductNonnegativeReal

export Mux, ConditionalScore, ConditionalSample, ToAssmts, LookupTable
export NonnegativeRealMultiplier, Theta, Step
export CPTSampleScore, MultiInputLookupTable
export ValuePasser, ValueBlocker

end