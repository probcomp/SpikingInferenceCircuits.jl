#=
This file provides support for circuit compilation of circuits from
the `ProbEstimates` module.
=#
using DiscreteIRTransforms, ProbEstimates

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
        to_labeled_cpts(d.latent_model, arg_domains),
        to_labeled_cpts(d.obs_model, obs_arg_domains(d, arg_domains)),
        to_labeled_cpts(d.proposal, (
                arg_domains...,
                get_ret_domain(d.obs_model, obs_arg_domains(d, arg_domains))
            )
        ),
        d.val_to_obs_choicemap,
        d.n_particles,
        d.args_for_compilation
    )
function to_indexed_cpts(d::PseudoMarginalizedDist, arg_domains)
    latent_model, _, _   = to_indexed_cpts(d.latent_model, arg_domains)
    obs_model, _, retbij = to_indexed_cpts(d.obs_model, obs_arg_domains(d, arg_domains))
    proposal, _, _ = to_indexed_cpts(
        d.proposal,
        (arg_domains..., get_ret_domain(d.obs_model, obs_arg_domains(d, arg_domains)))
    )

    return (
        PseudoMarginalizedDist{Int}(
            latent_model, obs_model, proposal, d.val_to_obs_choicemap, d.n_particles,
            (
                [FiniteDomain((length ∘ DiscreteIRTransforms.vals)(x)) for x in latent_domains(d)],
                d.args_for_compilation[2:4]...
            )
        ),
        Dict(:val => retbij),
        retbij
    )
end
is_cpts(d::PseudoMarginalizedDist) = is_cpts(d.latent_model) && is_cpts(d.obs_model) && is_cpts(d.proposal)
get_ret_domain(d::PseudoMarginalizedDist, arg_domains) =
    get_ret_domain(d.obs_model, (arg_domains..., latent_domains(d)...))
with_constant_inputs_at_indices(d::PseudoMarginalizedDist{R}, idx_val_pairs) where {R} =
    PseudoMarginalizedDist{R}(
        with_constant_inputs_at_indices(d.latent_model, idx_val_pairs),
        d.obs_model,
        # indices for proposal inputs are same as model inputs, since the proposal
        # gets the model inputs first (and then the return value)
        with_constant_inputs_at_indices(d.proposal, idx_val_pairs),
        d.val_to_obs_choicemap,
        d.n_particles,
        d.args_for_compilation
    )

gen_fn_circuit(d::PseudoMarginalizedDist, arg_domains, op) =
    pmdist_to_gen_fn_circuit(d, arg_domains, op)
