tuning_curve(cont, disc) = Gen.logpdf(normal, cont, disc, Yᶜ_STD())
tuning_curve_values(cont) = exp.([
    tuning_curve(cont, pos) for pos in Positions()
])
@gen (static) function _exact_init_proposal_contobs(yᶜₜ)
    yᵈₜ ~ ContinuousInvertingCat(tuning_curve_values(yᶜₜ))
    xₜ ~ Cat(init_posterior(yᵈₜ))
    vₜ ~ LCat(Vels())(unif(Vels()))
end
@gen (static) function _exact_step_proposal_contobs(xₜ₋₁, vₜ₋₁, yᵈₜ₋₁, yᶜₜ)
    yᵈₜ ~ ContinuousInvertingCat(tuning_curve_values(yᶜₜ))
    vₜ ~ LCat(Vels())(vel_step_posterior(xₜ₋₁, vₜ₋₁, yᵈₜ))
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
end

@gen (static) function _approx_step_proposal_contobs(xₜ₋₁, vₜ₋₁, yᵈₜ₋₁, yᶜₜ)
    yᵈₜ ~ ContinuousInvertingCat(tuning_curve_values(yᶜₜ))
    obs_prev_diff = yᵈₜ - xₜ₋₁
    vₜ ~ LCat(Vels())(v_t_dist(obs_prev_diff, vₜ₋₁))
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
end

exact_init_proposal_contobs = @compile_initial_proposal(_exact_init_proposal_contobs, 1)
exact_step_proposal_contobs = @compile_step_proposal(_exact_step_proposal_contobs, 3, 1)
approx_step_proposal_contobs = @compile_step_proposal(_approx_step_proposal_contobs, 3, 1)
@load_generated_functions()
