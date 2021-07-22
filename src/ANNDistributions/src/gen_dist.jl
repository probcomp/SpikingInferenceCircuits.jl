using Gen
using ProbEstimates: LCat

struct ANN_LCPT_Trace{T} <: Gen.Trace
    args::Tuple
    lcat_tr
    gf::GenerativeFunction{T}
end
Gen.get_args(tr::ANN_LCPT_Trace) = tr.args
Gen.get_retval(tr::ANN_LCPT_Trace) = get_retval(tr.lcat_tr)
Gen.get_choices(tr::ANN_LCPT_Trace) = get_choices(tr.lcat_tr)
Gen.get_score(tr::ANN_LCPT_Trace) = get_score(tr.lcat_tr)
Gen.get_gen_fn(tr::ANN_LCPT_Trace) = tr.gf
Gen.project(tr::ANN_LCPT_Trace, ::EmptySelection) = 0.

struct ANN_LCPT{T, A} <: Gen.GenerativeFunction{T, ANN_LCPT_Trace{T}}
    in_domains :: Tuple
    out_labels :: Vector{T}
    ann        :: A
end
ANN_LCPT(in_domains::Tuple, out_labels, ann) = ANN_LCPT(in_domains, collect(out_labels), ann)
get_probs(a::ANN_LCPT, assmt) = normalize(convert(Vector{Float64}, a.ann(assmt_to_onehots(a, assmt))))
assmt_to_onehots(a::ANN_LCPT, assmt) =
    ANNDistributions.assmt_to_onehots(labeled_assmt_to_indexed(a.in_domains, assmt), map(length, a.in_domains))
labeled_assmt_to_indexed(in_domains, assmt) = [val - first(dom) + 1 for (val, dom) in zip(assmt, in_domains)]

Gen.simulate(a::ANN_LCPT, assmt::Tuple) = ANN_LCPT_Trace(
    assmt,
    simulate(LCat(a.out_labels), (get_probs(a, assmt),)),
    a
)
Gen.generate(a::ANN_LCPT, assmt::Tuple, cm::Gen.ChoiceMap) = ANN_LCPT_Trace(
    assmt,
    generate(LCat(a.out_labels), (get_probs(a, assmt),), cm),
    a
)
function Gen.update(tr::ANN_LCPT_Trace, new_assmt::Tuple, diffs::Tuple, cm::Gen.ChoiceMap)
    oldprobs = get_probs(get_gen_fn(tr), get_args(tr))
    newprobs = get_probs(get_gen_fn(tr), new_assmt)
    diff = oldprobs == newprobs ? NoChange() : UnknownChange()
    new_subtr, weight, retdiff, discard = update(tr.lcpt_tr, (newprobs,), (diff,), cm)
    newtr = ANN_LCPT_Trace(new_assmt, new_subtr, get_gen_fn(tr))
    return (newtr, weight, retdiff, discard)
end

### Compilation Interface ###
# DiscreteIRTransforms.is_cpts(::ANN_LCPT) = true
# DiscreteIRTransforms.get_ret_domain(a::ANN_LCPT, arg_domains) =
#     DiscreteIRTransforms.EnumeratedDomain(a.out_labels)
# DiscreteIRTransforms.get_domain(a::ANN_LCPT, arg_domains) = DiscreteIRTransforms.get_ret_domain(a, arg_domains)
# # TODO: to indexed cpts

# struct ANNDistGenFn{Op} <: SIC.GenFn{Op}
#     ANN_LCPT
# end

# SpikingInferenceCircuits.gen_fn_circuit(d::ANN_LCPT, arg_domains, op) =
    