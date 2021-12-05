Yᶜ_STD() = 0.2

@gen (static) function initial_latent_model_contobs()
    xₜ ~ Cat(unif(Positions()))
    vₜ ~ LCat(Vels())(unif(Vels()))
    yᵈₜ ~ Cat(discretized_gaussian(xₜ, ObsStd(), Positions()))
    return (xₜ, vₜ, yᵈₜ)
end
@gen (static) function step_latent_model_contobs(xₜ₋₁, vₜ₋₁)
    vₜ ~ LCat(Vels())(
        (1 - SwitchProb()) * discretized_gaussian(vₜ₋₁, VelStepStd(), Vels()) +
             SwitchProb()  * unif(Vels())
    )
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
    yᵈₜ ~ Cat(discretized_gaussian(xₜ, ObsStd(), Positions()))
    return (xₜ, vₜ, yᵈₜ)
end

@gen (static) function obs_model_continuous(xₜ, vₜ, yᵈₜ)
    yᶜₜ = {:yᶜₜ => :val} ~ normal(yᵈₜ, Yᶜ_STD())
    return (yᶜₜ,)
end

model_contobs = @DynamicModel(initial_latent_model_contobs, step_latent_model_contobs, obs_model_continuous, 3)
@load_generated_functions()