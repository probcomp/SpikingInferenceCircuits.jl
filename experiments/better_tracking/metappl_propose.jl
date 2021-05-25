
# Generative function combiantor that overrides internal proposal with another
# generative function

# Not yet implemented:
# - update
# - project
# - choice_gradients
# - accumulate_param_gradients!

struct WithPseudoMargAuxTrace{U} <: Trace
    model_trace::Gen.Trace{U}
    gen_fn::Gen.GenerativeFunction
end

struct WithPseudoMargAux{T, U} <: Gen.GenerativeFunction{T, WithPseudoMargAuxTrace{U}}
    model::GenerativeFunction{T, U}
    proposal::GenerativeFunction
    n_particles::Int
end

Gen.get_args(tr::WithPseudoMargAuxTrace) = get_args(tr.model_trace)
Gen.get_retval(tr::WithPseudoMargAuxTrace) = get_retval(tr.model_trace)
Gen.get_choices(tr::WithPseudoMargAuxTrace) = choicemap((:ret, get_retval(tr)))
Gen.get_score(tr::WithPseudoMargAuxTrace) = # TODO

Gen.get_gen_fn(tr::WithPseudoMargAuxTrace) = tr.gen_fn

function Gen.project(tr::ReplaceProposalGFTrace, ::EmptySelection)
    return project(tr.model_trace, EmptySelection())
end

function Gen.simulate(gen_fn::WithPseudoMargAux, args::Tuple)
    tr = simulate(gen_fn.model, args)
    return WithPseudoMargAuxTrace(tr, gen_fn)
end

function Gen.propose(gen_fn::WithPseudoMargAux, args::Tuple)

end

# function generate(gen_fn::WithPseudoMargAux, args::Tuple, constraints::EmptyChoiceMap)


#     (proposed_choices, proposal_weight, _) = propose(gen_fn.proposal, (constraints, args...))
#     all_constraints = merge(proposed_choices, constraints)
#     new_tr, model_weight = generate(gen_fn.model, args, all_constraints)
#     @assert isapprox(model_weight, get_score(new_tr))
#     weight = model_weight - proposal_weight
#     return (ReplaceProposalGFTrace(new_tr, gen_fn), weight)
# end

# function regenerate(trace::ReplaceProposalGFTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
#     gen_fn = get_gen_fn(trace)
#     prev_args = get_args(trace)

#     # u <- create choice map u containing addresses from trace, except for those in selection
#     u = get_selected(get_choices(trace), complement(selection))

#     # then, run generate with that u to obtain new-trace t', and weight w = p(t'; x') / q(t; x, u')
#     (new_trace, p_weight) = generate(gen_fn, args, u)

#     # then, create choice map u' containing addresses from new-trace, except for those in selection
#     u_backward = get_selected(get_choices(new_trace), complement(selection))

#     # then, run generate on custom_q to obtain q(t; x, u')
#     (_, q_weight) = generate(gen_fn.proposal, (u_backward, prev_args...), get_choices(trace)) # NOTE there will be extra choices
    
#     # then, use get_score(trace) and subtracct it from the weight
#     weight = p_weight + q_weight - get_score(trace)

#     return (new_trace, weight, UnknownChange())
# end

# function override_internal_proposal(p, q)
#     return ReplaceProposalGF(p, q)
# end

# export override_internal_proposal