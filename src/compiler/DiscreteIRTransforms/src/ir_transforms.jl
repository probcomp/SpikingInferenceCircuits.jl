# module IRTransforms
using Revise
using Gen
using Gen: StaticIRGenerativeFunction
using Gen: StaticIRNode, RandomChoiceNode, JuliaNode, ArgumentNode, GenerativeFunctionCallNode
using Gen: Bernoulli, UniformDiscrete, Categorical
using Bijections
using Setfield: @set, setproperties

# import ..CPT, ..LabeledCPT

includet("ir_manipulation.jl")

## distribution specific: ##

get_domain(::Bernoulli, _) = EnumeratedDomain([true, false])
assmt_to_probs(::Bernoulli) = ((p,),) -> [p, 1 - p]

get_domain(::Gen.Categorical, arg_domains) = EnumeratedDomain(1:length(only(arg_domains)))
assmt_to_probs(::Gen.Categorical) = ((pvec,),) -> pvec

get_domain(::UniformDiscrete, (start_dom, end_dom)) = EnumeratedDomain(minimum(start_dom):maximum(end_dom))
assmt_to_probs(::UniformDiscrete) = ((min, max),) -> [1/(1 + max - min) for _=min:max]

get_domain(c::LabeledCPT, _) = EnumeratedDomain([c.output_values[i] for i=1:length(c.output_values)])

### main functions ###

includet("get_domains.jl")

includet("to_labeled_cpts.jl")
export to_labeled_cpts

includet("to_indexed_cpts.jl")

# end