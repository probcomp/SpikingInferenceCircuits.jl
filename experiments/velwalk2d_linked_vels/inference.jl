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
    vₜ ~ LCat(Vels2D())(unif(Vels2D()))
    (vx, vy) = vₜ
    vxₜ ~ LCat(Vels())(onehot(vx, Vels()))
    vyₜ ~ LCat(Vels())(onehot(vy, Vels()))
end

@gen (static) function _prior_step_proposal(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, obsx, obsy)
    vₜ ~ LCat(Vels2D())(vel_step_dist((vxₜ₋₁, vyₜ₋₁)))
    (vx, vy) = vₜ
    vxₜ ~ LCat(Vels())(onehot(vx, Vels()))
    vyₜ ~ LCat(Vels())(onehot(vy, Vels()))

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
    orig_typ = ProbEstimates.weight_type()
    ProbEstimates.use_perfect_weights!()

    logprobs, _ = enumeration_filter_init(initial_latent_model, obs_model,
        choicemap((:obsx => :val, obsx), (:obsy => :val, obsy)),
        Dict(
            (:xₜ => :val) => Positions(),
            (:yₜ => :val) => Positions(),
            # vel shouldn't matter, so keep constant to speed this up
            (:vxₜ => :val) => [0],
            (:vyₜ => :val) => [0],
            (:vₜ => :val) => [(0, 0)],
        )
    )
    
    ProbEstimates.reset_weights_to!(orig_typ)

    return reshape(
        sum(exp.(logprobs), dims=3),
        size(logprobs)[1:2]
     ) |> normalize
end

function vel_step_posterior(xₜ₋₁, yₜ₋₁, vₜ₋₁, obsx, obsy)
    orig_typ = ProbEstimates.weight_type()
    ProbEstimates.use_perfect_weights!()

    logprobs, _ = enumeration_filter_step(
        step_latent_model, obs_model,
        choicemap((:obsx => :val, obsx), (:obsy => :val, obsy)),
        Dict(
            (:xₜ => :val) => Positions(),
            (:yₜ => :val) => Positions(),
            (:vxₜ => :val) => Vels(),
            (:vyₜ => :val) => Vels(),
            (:vₜ => :val) => Vels2D(),
        ),
        [0.], [(xₜ₋₁, yₜ₋₁, vₜ₋₁[1], vₜ₋₁[2])], 4
    )

    ProbEstimates.reset_weights_to!(orig_typ)

    # return the probs for the different velocity values
    return reshape(
            sum(exp.(logprobs), dims=(1, 2, 3, 4)),
            (length(Vels2D()),)
     ) |> normalize
end
to_vect(v) = reshape(v, (:,))
xprobs(grid) = sum(grid, dims=2) |> normalize |> to_vect
yprobs(grid, x) = grid[x, :]     |> normalize |> to_vect

@gen (static) function _exact_init_proposal(obsx, obsy)
    probgrid = init_posterior(obsx, obsy)
    xₜ ~ Cat(xprobs(probgrid))
    yₜ ~ Cat(yprobs(probgrid, xₜ))
    vₜ ~ LCat(Vels2D())(unif(Vels2D()))
end
@gen (static) function _exact_step_proposal(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, obsx, obsy)
    probs = vel_step_posterior(xₜ₋₁, yₜ₋₁, (vxₜ₋₁, vyₜ₋₁), obsx, obsy)
    vₜ ~ LCat(Vels2D())(probs)
    (vxₜ, vyₜ) = vₜ
    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, Positions()))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, Positions()))
end
exact_init_proposal = @compile_initial_proposal(_exact_init_proposal, 2)
exact_step_proposal = @compile_step_proposal(_exact_step_proposal, 4, 2)
@load_generated_functions()

smc_exact_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, exact_step_proposal)

### More efficient step proposal (in terms of BN size) ###

function vel_dist_1d(obs_prev_diff, vₜ₋₁)
    unnormalized_prior_probs =
        (1 - SwitchProb()) * discretized_gaussian(vₜ₋₁, VelStepStd(), Vels()) +
             SwitchProb()  * unif(Vels())

    # This is not quite the exact obs probs, since we don't handle the boundary conditions.
    # (This would be exact if there were no boundaries.)
    errs = (first(Positions()) - last(Positions()) - last(Vels())):(
        last(Positions()) - first(Positions()) - first(Vels())
    )
    
    unnormalized_obs_probs  = [
        discretized_gaussian(0, ObsStd(), errs)[obs_prev_diff - vₜ + last(Positions()) + last(Vels())]
        for vₜ in Vels()
    ]
    unnormalized_probs = unnormalized_prior_probs .* unnormalized_obs_probs
    return normalize(unnormalized_probs)
end

@gen (static) function _approx_step_proposal(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, obsx, obsy)
    xobs_prev_diff = obsx - xₜ₋₁
    yobs_prev_diff = obsy - yₜ₋₁

    vxₜ ~ LCat(Vels())(vel_dist_1d(xobs_prev_diff, vxₜ₋₁))
    vyₜ ~ LCat(Vels())(vel_dist_1d(yobs_prev_diff, vyₜ₋₁))
    vₜ ~ LCat(Vels2D())(onehot((vxₜ, vyₜ), Vels2D()), Vels2D())

    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, Positions()))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, Positions()))
end

approx_step_proposal = @compile_step_proposal(_approx_step_proposal, 4, 2)
@load_generated_functions()

smc_approx_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, approx_step_proposal)

#=
# More expensive possible approximate proposal -- 

function approx_vel_dist(xobs_prev_diff, yobs_prev_diff, vₜ₋₁)
    unnormalized_prior_probs = [vel_step_dist(vₜ₋₁)[vₜ] for vₜ in Vels2D()]
    unnormalized_x_obs_probs = [
        discretized_gaussian(0, ObsStd(), errs)[xobs_prev_diff + vₜ[1] + last(Positions())]
        for vₜ in Vels2D()
    ]
    unnormalized_y_obs_probs = [
        discretized_gaussian(0, ObsStd(), errs)[yobs_prev_diff + vₜ[2] + last(Positions())]
        for vₜ in Vels2D()
    ]
    unnormalized_probs = unnormalized_prior_probs .* unnormalized_x_obs_probs .* unnormalized_y_obs_probs
    return normalize(unnormalized_probs)
end
@gen (static) function _approx_step_proposal(xₜ₋₁, yₜ₋₁, vₜ₋₁, obsx, obsy)
    xobs_prev_diff = obsx - xₜ₋₁
    yobs_prev_diff = obsy - yₜ₋₁
    vₜ ~ LCat(Vels())(approx_vel_dist(xobs_prev_diff, yobs_prev_diff, vₜ₋₁))
    (vxₜ, vyₜ) = vₜ
    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, Positions()))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, Positions()))
end
=#