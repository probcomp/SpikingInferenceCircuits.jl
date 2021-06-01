### Domains for discrete values ###
Xs = 1:40; Vels = -4:4; Energies = 1:30
HOME = 20 # home at the 20th x position

### Definitions of probability distributions & Utils ###
moving_away_from_home(vₜ₋₁, xₜ₋₁) = sign(vₜ₋₁) == sign(xₜ₋₁ - HOME)
prior_p_stop_tired(eₜ₋₁) = -exp(eₜ₋₁/10)
prior_p_stop_far(xₜ₋₁, vₜ₋₁) = moving_away_from_home(vₜ₋₁, xₜ₋₁) ?
                                    1-exp(-abs(xₜ₋₁-HOME)/10) : 0.
expected_energy(eₜ₋₁, vₜ) = eₜ₋₁ + (abs(vₜ) > 0 ? -abs(vₜ) : 2.)

@gen (static) function initial_model()
    vₜ ~ uniform_discrete(Vels)
    xₜ ~ uniform_discrete(Xs)
    eₜ ~ uniform_discrete(Energies)
end
@gen (static) function step_model(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    vₜ ~ pseudomarginalized_vel_dist(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    xₜ ~ categorical(maybe_one_off(xₜ₋₁ + vₜ, 0.6, Xs))
    eₜ ~ categorical(maybe_one_off
                  expected_energy(eₜ₋₁, vₜ), 0.5), Energies)
end
@gen (static) function obs_model(eₜ, vₜ, xₜ)
    obsₜ ~ categorical(discretized_gaussian(xₜ, 2.0, Xs))
end
dpp = DynamicalProbabilisticProgram(initial_model,
                                        step_model, obs_model)