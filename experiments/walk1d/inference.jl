@gen (static) function _prior_init_proposal(obs)
    xₜ ~ Cat(unif(Positions()))
    return (xₜ,)
end
@gen (static) function _prior_step_proposal(xₜ₋₁, obs)
    xₜ ~ Cat(discretized_gaussian(xₜ₋₁, StepStd(), Positions()))
    return (xₜ,)
end

prior_init_proposal = @compile_initial_proposal(_prior_init_proposal, 1)
prior_step_proposal = @compile_step_proposal(_prior_step_proposal, 1, 1)
@load_generated_functions()

function smc_from_prior(tr, n_particles)
    obss = get_dynamic_model_obs(tr)
    (unweighted_inferences, weighted_inferences) = dynamic_model_smc(
        get_gen_fn(tr), obss, cm -> (cm[:obs => :val],),
        prior_init_proposal, prior_step_proposal, n_particles
    )
    return (unweighted_inferences, weighted_inferences)
end