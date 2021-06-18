using Gen, Distributions
include("modeling_utils.jl")
include("model_hyperparams.jl")


InitialLatents() = (2, 2, 18, -2)
@gen (static) function initial_latent_model()
    return InitialLatents()
end
@gen (static) function step_latent_model(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁)
    vxₜ ~ labeled_categorical(Vels(), maybe_one_off(vxₜ₋₁, 0.3, Vels()))
    vyₜ ~ labeled_categorical(Vels(), maybe_one_off(vyₜ₋₁, 0.3, Vels()))

    exp_x = xₜ₋₁ + vxₜ
    exp_y = yₜ₋₁ + vyₜ
    xₜ ~ categorical(truncated_discretized_gaussian(exp_x, 1.0, Positions()))
    yₜ ~ categorical(truncated_discretized_gaussian(exp_y, 1.0, Positions()))
    return (xₜ, vxₜ, yₜ, vyₜ)
end
@gen (static) function obs_model(xₜ, vxₜ, yₜ, vyₜ)
    obsx ~ categorical(truncated_discretized_gaussian(xₜ, 4.0, Positions()))
    obsy ~ categorical(truncated_discretized_gaussian(yₜ, 4.0, Positions()))

    return (obsx, obsy)
end

### proposals
@gen (static) function initial_proposal(obs)
end
@gen (static) function step_proposal(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁, obsx, obsy)
    projected_x = truncate_value(xₜ₋₁ + vxₜ₋₁, Positions())
    mean_x = (obsx + projected_x)/2
    xₜ ~ categorical(
        truncated_discretized_gaussian(mean_x, 1.5, Positions()) # TODO: make disc gauss support non-integer mean
    )
    diff_x = xₜ - xₜ₋₁
    vxₜ ~ labeled_categorical(Vels(),
        truncated_discretized_gaussian(diff_x, 1.0, Vels())
    )

    projected_y = truncate_value(yₜ₋₁ + vyₜ₋₁, Positions())
    mean_y = (obsy + projected_y)/2
    yₜ ~ categorical(
        truncated_discretized_gaussian(mean_y, 1.5, Positions()) # TODO: make disc gauss support non-integer mean
    )
    diff_y = yₜ - yₜ₋₁
    vyₜ ~ labeled_categorical(Vels(),
        truncated_discretized_gaussian(diff_y, 1.0, Vels())
    )
end
