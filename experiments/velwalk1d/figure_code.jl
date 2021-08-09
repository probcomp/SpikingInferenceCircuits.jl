Vels = -3:3; Xs = 1:20
@gen (static) function step_latent_model(xₜ₋₁, vₜ₋₁)
   vₜ ~ velocity_step_model(vₜ₋₁)
   xₜ ~ Exactly(truncate(xₜ₋₁ + vₜ, Xs))
end
@gen (static) function obs_model(xₜ, vₜ)
   obsₜ ~ DiscretizedGaussian(xₜ, 2.0, Xs)
end

@gen (static) function step_proposal(xₜ₋₁, vₜ₋₁, obsₜ)
    obs_prev_diff = obsₜ - xₜ₋₁
    # VelProposal is compiled to a CPT; code shown in appendix
    vₜ ~ VelProposal(obs_prev_diff, vₜ₋₁)
    xₜ ~ Exactly(truncate(xₜ₋₁ + vₜ, Xs))
end

@gen (static) function vel_changepoint_model(vₜ₋₁)
   is_changepoint ~ Bernoulli(0.1)
end
@gen (static) function new_vel_model(vₜ₋₁, is_changepoint)
   vₜ ~ (  is_changepoint ? Uniform(Vels) :
       DiscretizedGaussian(vₜ₋₁, 0.5, Vels)  )
end
@gen (static) function changepoint_proposal(vₜ₋₁, vₜ)
   is_changepoint ~ Bernoulli( TODO )
end
velocity_step_model = PseudoMarginalizedDist(
   vel_changepoint_model, new_vel_model,
   is_changepoint_proposal
)
