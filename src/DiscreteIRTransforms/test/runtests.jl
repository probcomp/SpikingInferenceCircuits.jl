using Test
using Gen
using Bijections
using DiscreteIRTransforms
using DiscreteIRTransforms: get_ir, get_domains

normalize(vec) = vec / sum(vec)

include("simple_static_ir_transform.jl")
include("random_walk_ir_transform.jl")
include("map_ir_transform.jl")

include("inline_constants.jl")

# TODO: I think the way I convert LabeledCPT distributions to bijections may not be centralized;
# I should probably fix this