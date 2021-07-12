# TODO: refector so we don't repeat ourselves so much!

function smc(tr, n_particles, initprop, stepprop; always_resample=true)
    obss = get_dynamic_model_obs(tr)
    (unweighted_inferences, weighted_inferences) = dynamic_model_smc(
        get_gen_fn(tr), obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
        initprop, stepprop, n_particles; ess_threshold=(always_resample ? Inf : n_particles/2)
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
smc_from_prior_dont_always_resample(tr, n_particles) = smc(tr, n_particles, prior_init_proposal, prior_step_proposal, always_resample=false)

### Exact posterior ###
function init_posterior(obsx, obsy)
    logprobs, _ = enumeration_filter_init(initial_latent_model, obs_model,
        choicemap((:obsx => :val, obsx), (:obsy => :val, obsy)),
        Dict(
            (:xₜ => :val) => Positions(),
            (:yₜ => :val) => Positions(),
            # vel shouldn't matter, so keep constant to speed this up
            (:vxₜ => :val) => [0],
            (:vyₜ => :val) => [0]
        )
    )
    return reshape(
        sum(exp.(logprobs), dims=(3, 4)),
        size(logprobs)[1:2]
     ) |> normalize
end

function vel_step_posterior(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, obsx, obsy)
    logprobs, _ = enumeration_filter_step(
        step_latent_model, obs_model,
        choicemap((:obsx => :val, obsx), (:obsy => :val, obsy)),
        Dict(
            (:xₜ => :val) => Positions(),
            (:yₜ => :val) => Positions(),
            (:vxₜ => :val) => Vels(),
            (:vyₜ => :val) => Vels()
        ),
        [0.], [(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁)], 2
    )
    # return the probs for the different velocity values
    return reshape(
            sum(exp.(logprobs), dims=(1, 2)),
            size(logprobs)[3:4]
     ) |> normalize
end
to_vect(v) = reshape(v, (:,))
xprobs(grid) = sum(grid, dims=2) |> normalize |> to_vect
yprobs(grid, x) = grid[x, :]     |> normalize |> to_vect

@gen (static) function _exact_init_proposal(obsx, obsy)
    probgrid = init_posterior(obsx, obsy)
    xₜ ~ Cat(xprobs(probgrid))
    yₜ ~ Cat(yprobs(probgrid, xₜ))
    vxₜ ~ LCat(Vels())(unif(Vels()))
    vyₜ ~ LCat(Vels())(unif(Vels()))
end
@gen (static) function _exact_step_proposal(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, obsx, obsy)
    probs = vel_step_posterior(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, obsx, obsy)
    vxₜ ~ LCat(Vels())(xprobs(probs))
    vyₜ ~ LCat(Vels())(yprobs(probs, vxₜ - first(Vels()) + 1))
    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, Positions()))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, Positions()))
end
exact_init_proposal = @compile_initial_proposal(_exact_init_proposal, 2)
exact_step_proposal = @compile_step_proposal(_exact_step_proposal, 4, 2)
@load_generated_functions()

smc_exact_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, exact_step_proposal)
