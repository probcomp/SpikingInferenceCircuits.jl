using Gen, Distributions
using ProbEstimates: Cat, LCat

# Include some utilities for defining discrete probability distributions
includet("../../utils/modeling_utils.jl")
Positions() = 1:20; Vels() = [:unset, (-3:3)...];
Bools() = [true, false]
P_FAULTY_OBS() = 0.0
OBS_GAUSSIAN_STD() = 3.0

unif_set_vel() = normalize([v == :unset ? 0. : 1. for v in Vels()])
discard_value(val, pvec, vals) = [v == val ? 0. : p for (p, v) in zip(pvec, vals)] |> normalize

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    vxₜ ~ LCat(Vels())(onehot(:unset, Vels())) # velocity gets set during second timestep

    return (xₜ, vxₜ)
end
@gen (static) function step_latent_model(xₜ₋₁, vxₜ₋₁)
    vxₜ ~ LCat(Vels())(
        vxₜ₋₁ == :unset ?
                  unif_set_vel() :
                  onehot(vxₜ₋₁, Vels())
    )

    exp_x = xₜ₋₁ + vxₜ
    xₜ ~ Cat(maybe_one_off(exp_x, 0.3, Positions()))

    return (xₜ, vxₜ)
end

@gen (static) function obs_model_direct(xₜ, vxₜ)
    obsx ~ Cat(
            (1 - P_FAULTY_OBS()) * truncated_discretized_gaussian(xₜ, OBS_GAUSSIAN_STD(), Positions()) +
                P_FAULTY_OBS()  * unif(Positions())
    )

    return obsx
end

### proposals
@gen (static) function initial_proposal(obsx)
    xₜ ~ Cat(truncated_discretized_gaussian(obsx, 8., Positions()))
    vxₜ ~ LCat(Vels())(onehot(:unset, Vels()))
end
@gen (static) function step_proposal(xₜ₋₁, vxₜ₋₁, obsx)
    projected_x = vxₜ₋₁ == :unset ? :unset : truncate_value(xₜ₋₁ + vxₜ₋₁, Positions())
    mean_x = projected_x == :unset ? :unset : 0.5 * obsx + 0.5 * projected_x
    far_from_projection = projected_x == :unset ? :unset : abs(obsx - projected_x) > 3

    xₜ ~ Cat(
        mean_x == :unset ?
                  discretized_gaussian(obsx, 3.0, Positions()) :
                  truncate_dist_to_valrange(
                      discretized_gaussian(mean_x, 3.0, Positions()),
                      (projected_x - 1):(projected_x + 1),
                      Positions()
                  ) |> truncate
    )

    is_unset = vxₜ₋₁ == :unset
    vxₜ ~ LCat(Vels())(
        is_unset ?
            maybe_one_off(xₜ - xₜ₋₁, 0.3, Vels()) :
            onehot(vxₜ₋₁, Vels())
    )
end

vel_step_dist(vxₜ₋₁, diff_x) =
    let probs = maybe_one_off(vxₜ₋₁, 0.3, Vels()) .* maybe_one_off(diff_x, 0.5, Vels())
        isprobvec(probs) ? probs : maybe_one_off(vxₜ₋₁, 0.3, Vels())
    end |> normalize