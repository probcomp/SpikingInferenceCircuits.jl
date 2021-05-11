module SpikingInferenceCircuits

using Gen
using Circuits
using SpikingCircuits
using CPTs
using Distributions: ncategories

include("circuits/pulse_ir/pulse_ir.jl")
include("circuits/stochastic_digital_circuits/SDCs.jl")

using .SDCs
using .SDCs: CPTSample, CPTScore
include("circuits/generative_functions/gen_fn_circuits.jl")

export PulseIR, SDCs

export CPT, gen_fn_circuit, Propose, Generate, Assess
export FiniteDomain, IndexedProductDomain

include("circuits/inference/inference.jl")

export ISParticle

end
