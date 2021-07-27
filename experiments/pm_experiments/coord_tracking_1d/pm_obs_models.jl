using ProbEstimates: PseudoMarginalizedDist
### Observation models involving pseudo-marginalization
N_PM_PARTICLES() = 2

### model:
@gen (static) function faulty_obs_model(x)
    is_erroneous ~ LCat([true, false])([P_FAULTY_OBS(), 1 - P_FAULTY_OBS()])
    obs ~ Cat(
        is_erroneous ? unif(Positions()) : truncated_discretized_gaussian(x, OBS_GAUSSIAN_STD(), Positions())
    )
    return obs
end

### proposals for `is_erroneous`
@gen (static) function from_prior_is_erroneous_proposal(x, obs)
    is_erroneous ~ LCat([true, false])([P_FAULTY_OBS(), 1 - P_FAULTY_OBS()])
    obs ~ Cat(onehot(obs, Positions()))
end
function exact_is_err_probs(x, obs)
    likelihood_faulty_and_obs = P_FAULTY_OBS() * unif(Positions())[obs]
    likelihood_notfaulty_and_obs = (1 - P_FAULTY_OBS()) * truncated_discretized_gaussian(x, OBS_GAUSSIAN_STD(), Positions())[obs]
    @assert likelihood_faulty_and_obs + likelihood_notfaulty_and_obs > 0 "likelihood_faulty_and_obs = $likelihood_faulty_and_obs; likelihood_notfaulty_and_obs = $likelihood_notfaulty_and_obs"
    return [likelihood_faulty_and_obs, likelihood_notfaulty_and_obs] |> normalize
end
@gen (static) function exact_is_erroneous_proposal(x, obs)
    is_erroneous ~ LCat([true, false])(exact_is_err_probs(x, obs))
    obs ~ Cat(onehot(obs, Positions()))
end
# not exact, but better than from the prior; should also be a bit cheaper to implement
# than the exact one:
@gen (static) function intermediate_erroneous_proposal(x, obs)
    is_close = abs(x - obs) < 3.
    is_erroneous ~ LCat([true, false])(is_close ? [0.5, 0.5] : [0.2, 0.8])
    obs ~ Cat(onehot(obs, Positions()))
end
# TODO: exact marginalization? e.g. using ideas from RAVI

@gen (static) function obs_model_naive_pseudomarginalization(xₜ, vxₜ)
    obsx ~ PseudoMarginalizedDist(faulty_obs_model, from_prior_is_erroneous_proposal, N_PM_PARTICLES())(xₜ)
    return obsx
end
@gen (static) function obs_model_exact_pseudomarginalization(xₜ, vxₜ)
    obsx ~ PseudoMarginalizedDist(faulty_obs_model, exact_is_erroneous_proposal, N_PM_PARTICLES())(xₜ)
    return obsx
end
@gen (static) function obs_model_intermediate_pseudomarginalization(xₜ, vxₜ)
    obsx ~ PseudoMarginalizedDist(faulty_obs_model, intermediate_erroneous_proposal, N_PM_PARTICLES())(xₜ)
    return obsx
end
println("included pm file")