using Gen
using Distributions

include("modeling_utils.jl")
include("model_hyperparams.jl")

@gen (static) function step_model(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁)
    vxₜ ~ labeled_categorical(Vels(), maybe_one_off(vxₜ₋₁, 0.3, Vels()))
    vyₜ ~ labeled_categorical(Vels(), maybe_one_off(vyₜ₋₁, 0.3, Vels()))
    
    exp_x = xₜ₋₁ + vxₜ
    exp_y = yₜ₋₁ + vyₜ
    xₜ ~ categorical(maybe_one_off(exp_x, 0.6, Positions()))
    yₜ ~ categorical(maybe_one_off(exp_y, 0.6, Positions()))

    obsx ~ categorical(truncated_discretized_gaussian(xₜ, 2.0, Positions()))
    obsy ~ categorical(truncated_discretized_gaussian(yₜ, 2.0, Positions()))

    return (xₜ, vxₜ, yₜ, vyₜ, obsx, obsy)
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

@load_generated_functions()