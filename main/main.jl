module SNNs

using Distributions: Categorical, ncategories, probs

include("targets/types.jl")
include("value.jl")
include("component.jl")
include("targets/spiking/spiking.jl")

# Categorical Sampler
include("components/cat_sampler/abstract.jl")
include("components/cat_sampler/spiking.jl")

# Categorical Sampler which outputs probability of chosen Sampler
include("components/cat_sampler_with_prob/abstract.jl")

include("visualization/component_interface.jl")
include("visualization/animation_interface.jl")

end