# TODO: refector so we don't repeat ourselves so much!

function smc(tr, n_particles, initprop, stepprop)
    obss = get_dynamic_model_obs(tr)
    (unweighted_inferences, weighted_inferences) = dynamic_model_smc(
        get_gen_fn(tr), obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
        initprop, stepprop, n_particles
    )
    return (unweighted_inferences, weighted_inferences)
end

@gen (static) function _prior_init_proposal(obsx, obsy)
    xₜ ~ Cat(unif(Positions()))
    yₜ ~ Cat(unif(Positions()))
    vxₜ ~ LCat(Vels())(unif(Vels()))
    vyₜ ~ LCat(Vels())(unif(Vels()))
end

@gen (static) function _prior_step_proposal(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, obsx, obsy)
    vxₜ ~ LCat(Vels())(v_step_dist(vxₜ₋₁))
    vyₜ ~ LCat(Vels())(v_step_dist(vyₜ₋₁))

    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, Positions()))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, Positions()))
end

prior_init_proposal = @compile_initial_proposal(_prior_init_proposal, 2)
prior_step_proposal = @compile_step_proposal(_prior_step_proposal, 4, 2)
@load_generated_functions()

smc_from_prior(tr, n_particles) = smc(tr, n_particles, prior_init_proposal, prior_step_proposal)
