# TODO: refector so we don't repeat ourselves so much!

function smc(tr, n_particles, initprop, stepprop)
    obss = get_dynamic_model_obs(tr)
    (unweighted_inferences, weighted_inferences) = dynamic_model_smc(
        get_gen_fn(tr), obss, cm -> (cm[:obs => :val],),
        initprop, stepprop, n_particles
    )
    return (unweighted_inferences, weighted_inferences)
end

@gen (static) function _prior_init_proposal(obs)
    xₜ ~ Cat(unif(Positions()))
    vₜ ~ LCat(Vels())(unif(Vels()))
    return (xₜ, vₜ)
end
@gen (static) function _prior_step_proposal(xₜ₋₁, vₜ₋₁, obs)
    vₜ ~ LCat(Vels())(
        (1 - SwitchProb()) * discretized_gaussian(vₜ₋₁, VelStepStd(), Vels()) +
            SwitchProb()  * unif(Vels())
    )
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
    return (xₜ, vₜ)
end

prior_init_proposal = @compile_initial_proposal(_prior_init_proposal, 1)
prior_step_proposal = @compile_step_proposal(_prior_step_proposal, 2, 1)
@load_generated_functions()

smc_from_prior(tr, n_particles) = smc(tr, n_particles, prior_init_proposal, prior_step_proposal)

### Exact Proposal ###
to_vector(v) = reshape(v, (:,))
# TODO: better abstractions for these functions
function init_posterior(obs)
    logprobs, _ = enumeration_filter_init(initial_latent_model, obs_model,
        choicemap((:obs => :val, obs)),
        Dict((:xₜ => :val) => Positions(), (:vₜ => :val) => [0]) # vel shouldn't matter, so keep constant to speed this up # Vels())
    )
    return sum(exp.(logprobs), dims=2) |> normalize |> to_vector
end
function vel_step_posterior(xₜ₋₁, vₜ₋₁, obs)
    logprobs, _ = enumeration_filter_step(
        step_latent_model, obs_model,
        choicemap((:obs => :val, obs)),
        Dict((:xₜ => :val) => Positions(), (:vₜ => :val) => Vels()),
        [0.], [(xₜ₋₁, vₜ₋₁)]
    )
    # return the probs for the different velocity values
    return sum(exp.(logprobs), dims=1) |> normalize |> to_vector
end
@gen (static) function _exact_init_proposal(obs)
    xₜ ~ Cat(init_posterior(obs))
    vₜ ~ LCat(Vels())(unif(Vels()))
end
@gen (static) function _exact_step_proposal(xₜ₋₁, vₜ₋₁, obs)
    vₜ ~ LCat(Vels())(vel_step_posterior(xₜ₋₁, vₜ₋₁, obs))
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
end
exact_init_proposal = @compile_initial_proposal(_exact_init_proposal, 1)
exact_step_proposal = @compile_step_proposal(_exact_step_proposal, 2, 1)
@load_generated_functions()

smc_exact_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, exact_step_proposal)

### Gibbs Rejuvenation ###
@gen (static) function rejuv_proposal_init(tr)
    {:init => :latents} ~ _exact_init_proposal(getobs(tr)(0))
end
@gen (static) function rejuv_proposal_step(tr)
    t = get_args(tr)[1]
    {:steps => t => :latents} ~ _exact_step_proposal(
        getpos(tr)(t - 1), latents_choicemap(tr, t - 1)[:vₜ => :val], getobs(tr)(t)
    )
end
@load_generated_functions()

function prior_smc_exact_rejuv(tr, n_particles; is_gibbs=true)
    function rejuvenate(trace)
        if get_args(trace)[1] == 0
            newtrace, accepted = mh(trace, rejuv_proposal_init, ())
        else
            newtrace, accepted = mh(trace, rejuv_proposal_step, ())
        end
        if is_gibbs
            @assert accepted "Should be a gibbs move."
        end
        return newtrace
    end

    obss = get_dynamic_model_obs(tr)
    (unweighted_inferences, weighted_inferences) = dynamic_model_smc(
        get_gen_fn(tr), obss, cm -> (cm[:obs => :val],),
        prior_init_proposal, prior_step_proposal, n_particles;
        rejuvenate
    )
    return (unweighted_inferences, weighted_inferences)
end

### MH Rejuvenation ###
# TODO