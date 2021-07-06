DiscreteIRTransforms.is_cpts(::LCat) = false

# LCat / Cat should be compiled into CPTs, even though this is
# not a distribution
DiscreteIRTransforms.compile_to_primitive(::LCat) = true

DiscreteIRTransforms.get_ret_domain(l::LCat, arg_domains) =
    DiscreteIRTransforms.EnumeratedDomain(labels(l, first(only(arg_domains))))

DiscreteIRTransforms.get_domain(l::LCat, arg_domains) = DiscreteIRTransforms.get_ret_domain(l, arg_domains)
DiscreteIRTransforms.assmt_to_probs(::LCat) = ((pvec,),) -> pvec

import DiscreteIRTransforms: to_labeled_cpts, to_indexed_cpts, is_cpts, get_ret_domain, with_constant_inputs_at_indices
to_labeled_cpts(d::PseudoMarginalizedDist{R}, arg_domains) where {R} =
    PseudoMarginalizedDist{R}(
        to_labeled_cpts(d.model, arg_domains),
        to_labeled_cpts(d.proposal, (
                arg_domains...,
                get_ret_domain(d.model, arg_domains)
            )
        ),
        d.n_particles
    )
function to_indexed_cpts(d::PseudoMarginalizedDist, arg_domains)
    model, _, retbij = to_indexed_cpts(d.model, arg_domains)
    proposal, _, _ = to_indexed_cpts(
        d.proposal,
        (arg_domains..., get_ret_domain(d.model, arg_domains))
    )

    return (
        PseudoMarginalizedDist{Int}(model, proposal, d.n_particles),
        Dict(:val => retbij),
        retbij
    )
end
is_cpts(d::PseudoMarginalizedDist) = is_cpts(d.model) && is_cpts(d.proposal)
get_ret_domain(d::PseudoMarginalizedDist, arg_domains) = get_ret_domain(d.model, arg_domains)
with_constant_inputs_at_indices(d::PseudoMarginalizedDist{R}, idx_val_pairs) where {R} =
    PseudoMarginalizedDist{R}(
        with_constant_inputs_at_indices(d.model, idx_val_pairs),
        # indices for proposal inputs are same as model inputs, since the proposal
        # gets the model inputs first (and then the return value)
        with_constant_inputs_at_indices(d.proposal, idx_val_pairs),
        d.n_particles
    )