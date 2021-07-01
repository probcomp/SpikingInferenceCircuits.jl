# Include the library exposing `Cat` and `LCat`
includet("../../../src/ProbEstimates/ProbEstimates.jl")
using .ProbEstimates

# Include some utilities for defining discrete probability distributions
includet("../../utils/modeling_utils.jl")
Positions() = 1:20; Vels() = -3:3;
Bools() = [true, false]
P_FAULTY_OBS() = 0.25

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    vxₜ ~ LCat(Vels())(unif(Vels()))

    return (xₜ, vxₜ)
end
@gen (static) function step_latent_model(xₜ₋₁, vxₜ₋₁)
    vxₜ ~ LCat(Vels())(maybe_one_off(vxₜ₋₁, 0.3, Vels()))

    exp_x = xₜ₋₁ + vxₜ
    xₜ ~ Cat(maybe_one_off(exp_x, 0.3, Positions()))
    return (xₜ, vxₜ)
end
@gen (static) function obs_model(xₜ, vxₜ)
    obsx ~ Cat(
        (1 - P_FAULTY_OBS()) * truncated_discretized_gaussian(xₜ, 3.0, Positions()) +
             P_FAULTY_OBS()  * unif(Positions())
    )

    return obsx
end

### proposals
@gen (static) function initial_proposal(obsx)
    xₜ ~ Cat(truncated_discretized_gaussian(obsx, 3., Positions()))
    vxₜ ~ LCat(Vels())(unif(Vels()))
end
@gen (static) function step_proposal(xₜ₋₁, vxₜ₋₁, obsx)
    projected_x = truncate_value(xₜ₋₁ + vxₜ₋₁, Positions())
    mean_x = 0.7 * obsx + 0.3 * projected_x

    xₜ ~ Cat(
        # it is possible to be up to 2 away from the projected_x
        truncate_dist_to_valrange(
            discretized_gaussian(mean_x, 3.0, Positions()),
            (projected_x - 2):(projected_x + 2),
            Positions()
        ) |> truncate
    )
    
    diff_x = xₜ - xₜ₋₁
    vxₜ ~ LCat(Vels())(vel_step_dist(vxₜ₋₁, diff_x))
end

vel_step_dist(vxₜ₋₁, diff_x) =
    let probs = maybe_one_off(vxₜ₋₁, 0.3, Vels()) .* maybe_one_off(diff_x, 0.5, Vels())
        isprobvec(probs) ? probs : maybe_one_off(vxₜ₋₁, 0.3, Vels())
    end |> normalize