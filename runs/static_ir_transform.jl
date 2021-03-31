using Gen
using Bijections

includet("../src/cpt.jl")
includet("../src/labeled_cpt.jl")
includet("../src/compiler/static_ir_transforms.jl")
using .StaticIRTransforms

@gen (static) function foo(x::Bool)
    y ~ bernoulli(x ? 0.2 : 0.8)
    return y
end

# using PyCall
# @pyimport graphviz
# Gen.draw_graph(foo, graphviz, "test")

with_cpts = StaticIRTransforms.to_labeled_cpts(foo, [[true, false]])