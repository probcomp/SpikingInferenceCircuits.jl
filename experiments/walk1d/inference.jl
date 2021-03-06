function smc(tr, n_particles, initprop, stepprop)
    obss = get_dynamic_model_obs(tr)
    (unweighted_inferences, weighted_inferences) = dynamic_model_smc(
        get_gen_fn(tr), obss, cm -> (cm[:obs => :val],),
        initprop, stepprop, n_particles
    )
    return (unweighted_inferences, weighted_inferences)
end

### SMC from Prior ###

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

smc_from_prior(tr, n_particles) = smc(tr, n_particles, prior_init_proposal, prior_step_proposal)

### SMC with Exact Postrior as Proposal ###
function init_posterior(obs)
    orig_typ = ProbEstimates.weight_type()
    ProbEstimates.use_perfect_weights!()
    logprobs, _ = enumeration_filter_init(initial_latent_model, obs_model, choicemap((:obs => :val, obs)), Dict((:xₜ => :val) => Positions()))
    ProbEstimates.reset_weights_to!(orig_typ)

    return exp.(logprobs) |> normalize
end
function step_posterior(xₜ₋₁, obs)
    orig_typ = ProbEstimates.weight_type()
    ProbEstimates.use_perfect_weights!()
    logprobs, _ = enumeration_filter_step(
        step_latent_model, obs_model,
        choicemap((:obs => :val, obs)), Dict((:xₜ => :val) => Positions()),
        [0.], [(xₜ₋₁,)]
    )
    ProbEstimates.reset_weights_to!(orig_typ)

    return exp.(logprobs) |> normalize
end
@gen (static) function _exact_init_proposal(obs)
    xₜ ~ Cat(init_posterior(obs))
    return (xₜ,)
end
@gen (static) function _exact_step_proposal(xₜ₋₁, obs)
    xₜ ~ Cat(step_posterior(xₜ₋₁, obs))
    return (xₜ,)
end
exact_init_proposal = @compile_initial_proposal(_exact_init_proposal, 1)
exact_step_proposal = @compile_step_proposal(_exact_step_proposal, 1, 1)
@load_generated_functions()

smc_exact_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, exact_step_proposal)

### Rejuvenation using exact posterior ##
# Rejuvenate last x value
@gen (static) function rejuv_proposal_init(tr)
    {:init => :latents} ~ _exact_init_proposal(getobs(tr)(0))
end
@gen (static) function rejuv_proposal_step(tr)
    t = get_args(tr)[1]
    {:steps => t => :latents} ~ _exact_step_proposal(getpos(tr)(t - 1), getobs(tr)(t))
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