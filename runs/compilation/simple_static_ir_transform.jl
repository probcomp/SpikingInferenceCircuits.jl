using Test

using Gen
using Bijections

includet("../../src/cpt.jl")
includet("../../src/labeled_cpt.jl")
includet("../../src/compiler/ir_transforms/ir_transforms.jl")
# using .IRTransforms

@gen (static) function foo(x::Bool)
    a = !x
    y ~ bernoulli([0.0, 1.0][a ? 1 : 2])
    return y
end

# using PyCall
# @pyimport graphviz
# Gen.draw_graph(foo, graphviz, "test")

input_domains = [EnumeratedDomain([true, false])]

with_lcpts = to_labeled_cpts(foo, input_domains)

(with_cpts, bijs) = to_indexed_cpts(foo, input_domains)
@load_generated_functions()

og_domains = get_domains(get_ir(foo).nodes, input_domains)
new_domains = get_domains(get_ir(with_lcpts).nodes, input_domains)

@test all(haskey(og_domains, name) && og_domains[name] == new_dom for (name, new_dom) in new_domains)

@test with_lcpts(true) == true
@test with_lcpts(false) == false
@test with_cpts(1) == 1
@test with_cpts(2) == 2

# TODO: I think a bug has arisen in which the same value is indexed differently in different parts of the code!