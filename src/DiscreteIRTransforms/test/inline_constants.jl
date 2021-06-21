# compiling away constant CPT inputs
cpt = CPT(
    [
        [[0.5, 0.5]] [[0.5, 0.5]];
        [[0.2, 0.8]] [[0.8, 0.2]]
    ]
)
cpt2 = DiscreteIRTransforms.with_constant_inputs_at_indices(cpt, [(1, 2)])
random(cpt2, 1)
@test Gen.logpdf(cpt2, 1, 1) == Gen.logpdf(cpt, 1, 2, 1)

function get_different_probs()
    probs = Array{Vector{Float64}, 3}(undef, 2, 2, 2)
    for x=1:2
        for y=1:2
            for z=1:2
                p = 1/2^x * 1/3^y * 1/5^z
                probs[x, y, z] = [p, 1-p]
            end
        end
    end
    return probs
end
cpt3 = CPT(get_different_probs())
cpt4 = DiscreteIRTransforms.with_constant_inputs_at_indices(cpt3, [(1, 2), (3, 1)])
random(cpt4, 2)
@test Gen.logpdf(cpt4, 1, 1) == Gen.logpdf(cpt3, 1, 2, 1, 1)

# compiling away constant LCPT inputs
lcpt = LabeledCPT{Bool}(
    [[true, false], [true, false]],
    [true, false],
    ((fst, snd),) -> (
        fst  && snd  ? [0.5, 0.5] :
        fst  && !snd ? [0.4, 0.6] :
        !fst && snd  ? [0.2, 0.8] :
                       [0.8, 0.2]
    )
)
lcpt2 = DiscreteIRTransforms.with_constant_inputs_at_indices(lcpt, [(1, false)])
random(lcpt2, true)
@test Gen.logpdf(lcpt2, false, true) == Gen.logpdf(lcpt, false, false, true)

# Inline constants simple

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

# Remove constant input node
@gen (static) function foo2(x, y)
    b ~ bernoulli(x > y ? 0.2 : 0.8)
    return b
end
@load_generated_functions()
lcpts = to_labeled_cpts(foo2, (EnumeratedDomain(1:4), EnumeratedDomain([2])))
@load_generated_functions()
inlined = DiscreteIRTransforms.with_constant_inputs_at_indices(lcpts, [(2, 2)])
@load_generated_functions()
@test length(get_ir(inlined).arg_nodes) == 1
