### Domains for discrete values ###
Xs = 1:40; Vels = -4:4; Energies = 1:30
HOME = 20 # home at the 20th x position

### Definitions of probability distributions ###
moving_away_from_home(vₜ₋₁, xₜ₋₁) = sign(vₜ₋₁) == sign(xₜ₋₁ - HOME)
prior_p_stop_tired(eₜ₋₁) = -exp(eₜ₋₁/10)
prior_p_stop_far(xₜ₋₁, vₜ₋₁) = moving_away_from_home(vₜ₋₁, xₜ₋₁) ?
                                    1-exp(-abs(xₜ₋₁-HOME)/10) : 0.
prop_p_stop_far(is_stopped, vₜ₋₁, xₜ₋₁) = !is_stopped ? 0. :
                       moving_away_from_home(vₜ₋₁, xₜ₋₁) ? 0.5 : 0.
prop_p_stop_tired(is_stopped, already_stopped, eₜ₋₁) =
    !is_stopped ? 0. : already_stopped ? prior_p_stop_tired(eₜ₋₁) : 0.6
expected_energy(eₜ₋₁, vₜ) = eₜ₋₁ + (abs(vₜ) > 0 ? -abs(vₜ) : 2.)

### (i) Dynamical Probabilistic Program, using Pseudo-Marginalized Velocity Model ###
@gen (static) function initial_model()
    vₜ ~ uniform_discrete(Vels)
    xₜ ~ uniform_discrete(Xs)
    eₜ ~ uniform_discrete(Energies)
end
@gen (static) function step_model(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    vₜ ~ pseudomarginalized_vel_dist(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    xₜ ~ categorical(maybe_one_off(xₜ₋₁ + vₜ, 0.6, Xs))
    eₜ ~ categorical(maybe_one_off(expected_energy(eₜ₋₁, vₜ), 0.5, Energies()))
end
@gen (static) function obs_model(eₜ, vₜ, xₜ)
    obsₜ ~ categorical(discretized_gaussian(xₜ, 2.0, Xs))
end
dpp = DynamicalProbabilisticProgram(initial_model, step_model, obs_model)

### (ii) Pseudo-Marginalized Velocity Model ###
@gen (static) function vel_step(eₜ₋₁, vₜ₋₁, xₜ₋₁)
    stop_because_tired ~ bernoulli(prior_p_stop_tired(eₜ₋₁))
    stop_because_far ~ bernoulli(prior_p_stop_far(xₜ₋₁, vₜ₋₁))
    vₜ ~ LabeledCategorical(Vels)(
        stop_because_tired || stop_because_far ?
                onehot(0, Vels) : # if stopping, velocity = 0
                # else, w.p. 0.8, change velocity by 1 or 2
                maybe_one_or_two_off(vₜ₋₁, 0.8, Vels)        )
end
# Proposal used to marginalize `stop_because_tired` and `stop_because_far`
@gen (static) function vel_pseudomarginal_proposal(vₜ, eₜ₋₁, vₜ₋₁, xₜ₋₁)
    stop_because_far ~ bernoulli(prop_p_stop_far(vₜ == 0, vₜ₋₁, xₜ₋₁))
    stop_because_tired ~ bernoulli(prop_p_stop_tired(vₜ == 0, stop_because_far, eₜ₋₁))
end
pseudomarginalized_vel_dist = PseudoMarginalize(vel_step, vel_pseudomarginal_proposal,
                                                          N_aux_particles=2)

@gen (static) function initial_proposal(obs₀)
    xₜ ~ categorical(discretized_gaussian(obs₀, 2.0, Xs))
    vₜ ~ uniform_discrete(Vels)
    eₜ ~ uniform_discrete(Energies)
end
@gen function step_proposal(eₜ₋₁, vₜ₋₁, xₜ₋₁, obsₜ)
    xₜ ~ categorical(discretized_gaussian(obsₜ, 2.0, Xs))
    vₜ ~ categorical(maybe_one_or_two_off(xₜ - xₜ₋₁, 0.5, Vels))
    eₜ ~ categorical(maybe_one_off(expected_e(eₜ₋₁, vₜ), .5, Energies))
end
pm_smc_circuit = Neural_PM_SMC_Circuit(dpp,
                       initial_proposal, step_proposal, n_particles=10)


