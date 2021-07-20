module SpikingInferenceCircuits
using Gen
using Circuits
using SpikingCircuits
using CPTs
using DiscreteIRTransforms
using Distributions: ncategories
using Setfield: @set

include("circuits/pulse_ir/pulse_ir.jl")
include("circuits/stochastic_digital_circuits/SDCs.jl")

using .SDCs
using .SDCs: CPTSample, CPTScore
include("circuits/generative_functions/gen_fn_circuits.jl")

export PulseIR, SDCs

export CPT, gen_fn_circuit, Propose, Generate, Assess
export FiniteDomain, IndexedProductDomain
export GenFnWithInputDomains

include("circuits/inference/is_particle.jl")
include("circuits/inference/mh.jl")
include("circuits/inference/rejuvenated_is_particle.jl")
include("circuits/inference/resample.jl")
include("circuits/inference/smc.jl")

export ISParticle
export SMCStep, RecurrentSMCStep, SMC

end
