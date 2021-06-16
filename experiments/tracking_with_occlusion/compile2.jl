using Gen
using Distributions
using DiscreteIRTransforms
using CPTs
using SpikingInferenceCircuits
using Revise

@gen (static) function foo(x)
    y = 2
    b ~ bernoulli(x > y ? 0.2 : 0.8)
    return b
end
@load_generated_functions()

lcpts = to_labeled_cpts(foo, (EnumeratedDomain(1:4),))
@load_generated_functions()

inlined = inline_constant_nodes(lcpts)
@load_generated_functions()
@test length(get_ir(inlined).nodes) == 2 # b, return

