Vals() = 1:20; Bools() = [true, false];
@gen (static) function noise_model(true_value)
    is_error ~ LCat(Bools())([0.2, 0.8])
    obs ~ Cat(
        is_error ? unif(Vals()) : maybe_one_off(true_value, 0.5, Vals())
    )
    return obs
end
@gen (static) function proposal(true_value, obs)
    is_error ~ LCat(Bools())([0.5, 0.5])
    obs ~ Cat(onehot(obs, Vals()))
end
@load_generated_functions()

true_noise_prob(true_value, obs) = (
        # p_not_error * p_obs_given_not_error
        0.8 * maybe_one_off(true_value, 0.5, Vals())[obs] +
        # p_is_error * p_obs_given_error
        0.2 * unif(Vals())[obs]
    )

obs_model = PseudoMarginalizedDist(noise_model, proposal, 100_000)
ProbEstimates.use_noisy_weights!()
# for _=1:10
#     proposed_choices, weight, ret = propose(obs_model, (10,))
#     @test proposed_choices[:val] == ret
#     recip_prob_est = exp(-weight)
#     @test abs(1/true_noise_prob(10, ret) - recip_prob_est) < 1e-1/true_noise_prob(10, ret)
# end

for _=1:10
    ret = obs_model(10)
    weight, _ = assess(obs_model, (10,), choicemap((:val, ret)))
    prob_est = exp(weight)
    @test abs(prob_est - true_noise_prob(10, ret)) < 1e-2
end