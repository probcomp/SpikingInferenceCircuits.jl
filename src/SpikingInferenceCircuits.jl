module SpikingInferenceCircuits
using Gen
using Circuits
using SpikingCircuits

include("cpt.jl")
include("value_types.jl")

include("components/mux/mux.jl")
include("components/ipoisson_gated_repeater.jl")
include("components/mux/int_poisson_mux.jl")
include("components/cvb.jl")
include("components/conditional_sample_score/abstract.jl")
include("components/conditional_sample_score/spiking.jl")
include("components/thresholded_spike_counter.jl")
include("components/to_assmts/abstract.jl")
include("components/to_assmts/spiking.jl")
include("components/cpt_sample_score/abstract.jl")
include("components/cpt_sample_score/spiking.jl")
include("components/real_multiplication/abstract.jl")
include("components/real_multiplication/rate_multiplier.jl")

include("compiler/compiler.jl")

export CPT, propose_circuit

end