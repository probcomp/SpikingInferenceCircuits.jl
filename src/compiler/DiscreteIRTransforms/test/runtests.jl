using Test
using Gen
using Bijections
using DiscreteIRTransforms
using DiscreteIRTransforms: get_ir, get_domains

normalize(vec) = vec / sum(vec)

# include("simple_static_ir_transform.jl")
# include("random_walk_ir_transform.jl")
include("map_ir_transform.jl")