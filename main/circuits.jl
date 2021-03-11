module Circuits

using Distributions: Categorical, ncategories, probs

include("utils.jl")

include("targets/types.jl")
include("value.jl")
include("component.jl")
include("targets/spiking/spiking.jl")

# Categorical Sampler
include("components/cat_sampler/abstract.jl")
include("components/cat_sampler/spiking.jl")

# Categorical Sampler which outputs probability of chosen Sampler
include("components/cat_sampler_with_prob/abstract.jl")
include("components/cat_sampler_with_prob/spiking.jl")

# Exporting to web visualization format
include("visualization/circuit_visualization/component_interface.jl")
include("visualization/circuit_visualization/animation_interface.jl")

end