using Gen, Distributions
include("../../src/DynamicModels.jl")
using .DynamicModels

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
    xₜ ~ categorical(maybe_one_off(exp_x, 0.3, Positions()))
    yₜ ~ categorical(maybe_one_off(exp_y, 0.3, Positions()))
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
    mean_x = 0.5 * obsx + 0.5 * projected_x

    xₜ ~ categorical(
        # it is possible to be up to 2 away from the projected_x
        truncate_dist_to_valrange(
            discretized_gaussian(mean_x, 2.5, Positions()),
            (projected_x - 2):(projected_x + 2),
            Positions()
        ) |> truncate
    )
    
    diff_x = xₜ - xₜ₋₁
    vxₜ ~ labeled_categorical(Vels(),
        vel_step_dist(vxₜ₋₁, diff_x)
        # err_if_not_probvec(normalize(
               
        #     ),   "vxₜ₋₁ = $vxₜ₋₁; xₜ = $xₜ; xₜ₋₁ = $xₜ₋₁; obsx = $obsx"
        # )
    )

    projected_y = truncate_value(yₜ₋₁ + vyₜ₋₁, Positions())
    mean_y = (obsy + projected_y)/2
    yₜ ~ categorical(
        truncate_dist_to_valrange(
            discretized_gaussian(mean_y, 2.5, Positions()),
            (projected_y - 2):(projected_y + 2),
            Positions()
        ) |> truncate
    )
    diff_y = yₜ - yₜ₋₁
    vyₜ ~ labeled_categorical(Vels(), vel_step_dist(vyₜ₋₁, diff_y))

        # err_if_not_probvec(
        #     normalize(
        #         Base.Broadcast.BroadcastFunction(*)(
        #             maybe_one_off(vyₜ₋₁, 0.3, Vels()),
        #             maybe_one_off(diff_y, 0.5, Vels())
        #         )
        #     ),
        #     "vyₜ₋₁ = $vyₜ₋₁; yₜ = $yₜ; yₜ₋₁ = $yₜ₋₁; obsy = $obsy"
        # )
end

err_if_not_probvec(pvec, errmsg) =
    if isprobvec(pvec)
        pvec
    else
        error(errmsg)
    end

vel_step_dist(vxₜ₋₁, diff_x) =
    let probs = maybe_one_off(vxₜ₋₁, 0.3, Vels()) .* maybe_one_off(diff_x, 0.5, Vels())
        isprobvec(probs) ? probs : maybe_one_off(vxₜ₋₁, 0.3, Vels())
    end |> normalize
