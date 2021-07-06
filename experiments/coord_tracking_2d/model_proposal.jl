using Gen, Distributions
# Include the library exposing `Cat` and `LCat`
using ProbEstimates

# Include some utilities for defining discrete probability distributions
includet("../utils/modeling_utils.jl")
Positions() = 1:8; Vels() = [:unset, (-1:1)...];
Bools() = [true, false]

unif_set_vel() = normalize([v == :unset ? 0. : 1. for v in Vels()])

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    yₜ ~ Cat(unif(Positions()))

    # velocity gets set during second timestep
    vxₜ ~ LCat(Vels())(onehot(:unset, Vels()))
    vyₜ ~ LCat(Vels())(onehot(:unset, Vels()))

    return (xₜ, vxₜ, yₜ, vyₜ)
end
@gen (static) function step_latent_model(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁)
    vxₜ ~ LCat(Vels())(
        vxₜ₋₁ == :unset ?
                unif_set_vel() :
                maybe_one_off(vxₜ₋₁, 0.1, Vels())
    )
    vyₜ ~ LCat(Vels())(
        vyₜ₋₁ == :unset ?
                  unif_set_vel() :
                  maybe_one_off(vyₜ₋₁, 0.1, Vels())
    )

    exp_x = xₜ₋₁ + vxₜ
    exp_y = yₜ₋₁ + vyₜ
    xₜ ~ Cat(maybe_one_off(exp_x, 0.0, Positions()))
    yₜ ~ Cat(maybe_one_off(exp_y, 0.0, Positions()))
    return (xₜ, vxₜ, yₜ, vyₜ)
end
@gen (static) function obs_model(xₜ, vxₜ, yₜ, vyₜ)
    obsx ~ Cat(posdist(xₜ))
    obsy ~ Cat(posdist(yₜ))

    return (obsx, obsy)
end

posdist(x) = 
    0.5 * onehot(x, Positions()) +
    0.5 * truncated_discretized_gaussian(x, 2.0, Positions())

### proposals
@gen (static) function initial_proposal(obsx, obsy)
    xₜ ~ Cat(discretized_gaussian(obsx, 0.5, Positions()))
    yₜ ~ Cat(discretized_gaussian(obsy, 0.5, Positions()))

    vxₜ ~ LCat(Vels())(onehot(:unset, Vels()))
    vyₜ ~ LCat(Vels())(onehot(:unset, Vels()))
end
@gen (static) function step_proposal(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁, obsx, obsy)
    projected_x = vxₜ₋₁ == :unset ? :unset : truncate_value(xₜ₋₁ + vxₜ₋₁, Positions())
    xₜ ~ Cat(
        projected_x == :unset ?
            posdist(obsx) : 
            truncate_dist_to_valrange(step_pos_dist(obsx, projected_x), (projected_x - 1):(projected_x + 1), Positions())
    )
    
    diff_x = xₜ - xₜ₋₁
    vxₜ ~ LCat(Vels())(vel_step_dist(vxₜ₋₁, diff_x))

    projected_y = vyₜ₋₁ == :unset ? :unset : truncate_value(yₜ₋₁ + vyₜ₋₁, Positions())
    yₜ ~ Cat(
        projected_y == :unset ?
            posdist(obsy) :
            truncate_dist_to_valrange(step_pos_dist(obsy, projected_y), (projected_y - 1):(projected_y + 1), Positions())
    )
    diff_y = yₜ - yₜ₋₁
    vyₜ ~ LCat(Vels())(vel_step_dist(vyₜ₋₁, diff_y))
end

# vel_step_dist(vxₜ₋₁, diff_x) =
#     vxₜ₋₁ == :unset ? maybe_one_off(diff_x, 0.1, Vels()) :
#     let probs = maybe_one_off(vxₜ₋₁, 0.2, Vels()) .* maybe_one_off(diff_x, 0.2, Vels())
#         isprobvec(probs) ? probs : maybe_one_off(vxₜ₋₁, 0.3, Vels())
#     end |> normalize
vel_step_dist(_, diff_x) = onehot(diff_x, Vels())

step_pos_dist(obsx, projected_x) =
    0.5 * onehot(projected_x, Positions()) +
    0.5 * discretized_gaussian((obsx + projected_x)/2, 1.5, Positions())