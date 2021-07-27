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
to_vect(v) = reshape(v, (:,))
# TODO: better abstractions for these functions
function init_posterior(obs)
    orig_typ = ProbEstimates.weight_type()
    ProbEstimates.use_perfect_weights!()
    logprobs, _ = enumeration_filter_init(initial_latent_model, obs_model,
        choicemap((:obs => :val, obs)),
        Dict((:xₜ => :val) => Positions(), (:vₜ => :val) => [0]) # vel shouldn't matter, so keep constant to speed this up # Vels())
    )
    ProbEstimates.reset_weights_to!(orig_typ)
    return sum(exp.(logprobs), dims=2) |> normalize |> to_vect
end
function vel_step_posterior(xₜ₋₁, vₜ₋₁, obs)
    orig_typ = ProbEstimates.weight_type()
    ProbEstimates.use_perfect_weights!()
    logprobs, _ = enumeration_filter_step(
        step_latent_model, obs_model,
        choicemap((:obs => :val, obs)),
        Dict((:xₜ => :val) => Positions(), (:vₜ => :val) => Vels()),
        [0.], [(xₜ₋₁, vₜ₋₁)]
        )
    ProbEstimates.reset_weights_to!(orig_typ)
    # return the probs for the different velocity values
    return sum(exp.(logprobs), dims=1) |> normalize |> to_vect
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

### Approximate step proposal, written with branching factor of 2 (rather than 3) ###
function v_t_dist(obs_prev_diff, vₜ₋₁)
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
kl(pv1, pv2) = sum(
    p1 * log(p1/p2) for (p1, p2) in zip(pv1, pv2)
)
@gen (static) function _approx_step_proposal(xₜ₋₁, vₜ₋₁, obs)
    obs_prev_diff = obs - xₜ₋₁
    vₜ ~ LCat(Vels())(v_t_dist(obs_prev_diff, vₜ₋₁))
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
end
approx_step_proposal = @compile_step_proposal(_approx_step_proposal, 2, 1)
@load_generated_functions()
smc_approx_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, approx_step_proposal)

### Gibbs Rejuvenation ###
@gen (static) function rejuv_proposal_init(tr)
    {:init => :latents} ~ _exact_init_proposal(obs_choicemap(tr, 0)[:obs => :val])
end
@gen (static) function rejuv_proposal_step(tr)
    t = get_args(tr)[1]
    {:steps => t => :latents} ~ _exact_step_proposal(
        latents_choicemap(tr, t - 1)[:xₜ => :val],
        latents_choicemap(tr, t - 1)[:vₜ => :val],
        obs_choicemap(tr, t)[:obs => :val]
    )
end
@load_generated_functions()

function gibbs(
    trace, proposal::GenerativeFunction, proposal_args::Tuple;
    # check the MH score iff we are using perfect weights;
    # we expect imperfect scores otherwise
    check_score=(ProbEstimates.weight_type() === :perfect)
    )
    # TODO add a round trip check
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = update(trace,
    model_args, argdiffs, fwd_choices
    )
    if check_score
        proposal_args_backward = (new_trace, proposal_args...,)
        (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)
        alpha = weight - fwd_weight + bwd_weight
        @assert abs(alpha) < 1e-4 "Is not a gibbs move."
    end
    return (new_trace, true)
end
mcmc_fn(is_gibbs) = is_gibbs ? gibbs : Gen.mh

function prior_smc_exact_rejuv(tr, n_particles; is_gibbs=true)
    function rejuvenate(trace)
        if get_args(trace)[1] == 0
            newtrace, accepted = mcmc_fn(is_gibbs)(trace, rejuv_proposal_init, ())
        else
            newtrace, accepted = mcmc_fn(is_gibbs)(trace, rejuv_proposal_step, ())
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