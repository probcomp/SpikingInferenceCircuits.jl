using Gen
using Bijections

includet("../src/cpt.jl")
includet("../src/labeled_cpt.jl")
includet("../src/compiler/ir_transforms/ir_transforms.jl")
# using .IRTransforms

@gen (static) function foo(x::Bool)
    y ~ bernoulli(x ? 0.2 : 0.8)
    return y
end

# using PyCall
# @pyimport graphviz
# Gen.draw_graph(foo, graphviz, "test")

(with_cpts, bijs) = to_indexed_cpts(Gen.get_ir(typeof(foo)), [[true, false]])