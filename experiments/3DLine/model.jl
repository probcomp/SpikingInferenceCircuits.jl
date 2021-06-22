using Gen
using Distributions


# commit test

include("../../src/ProbEstimates/ProbEstimates.jl")
ProbEstimates.use_perfect_weights!()
using .ProbEstimates: Cat, LCat

include("model_utils.jl")
include("model_hyperparams.jl")

@gen (static) function initial_model()
    x = 5
    y = 0
    height = 5
    moving_in_depth = true
    r = Int(round(sqrt(x^2 + y^2 + height^2)))
    v = 2
    
    return (moving_in_depth, v, height, x, y, r)
end

# x = back and forth
# y = left and right
# z = up and down (held constant in this model)
@gen (static) function step_model(moving_in_depthₜ₋₁, vₜ₋₁, heightₜ₋₁, xₜ₋₁, yₜ₋₁, rₜ₋₁)
    # TODO: experiment with 0.9 : 0.1 instead of 1.0 : 0.0
    moving_in_depthₜ ~ bernoulli(moving_in_depthₜ₋₁ ? 1.0 : 0.0)

    vₜ ~ LCat(Vels())(maybe_one_off(vₜ₋₁, 0.2, Vels()))

    heightₜ ~ Cat(moving_in_depthₜ ? onehot(heightₜ₋₁, Heights()) : maybe_one_off(heightₜ₋₁ - vₜ, 0.2, Heights()))
    xₜ ~ Cat(moving_in_depthₜ ? maybe_one_off(xₜ₋₁ + vₜ,  0.2, Xs()) : onehot(xₜ₋₁, Xs()))
    yₜ ~ Cat(maybe_one_off(yₜ₋₁ + 0.2, Ys()))

    # Here: a stochastic mapping from (x, y, h) -> (r, θ, ϕ)
    # TODO: make the probabilities perfectly correspond to the volume of overlap
    # between xyh voxels and rθϕ voxels.
    # For now: just use dimension-wise discretized Gaussians.
    # # rθϕ ~ LCat(RΘΦs())(xyh_to_azalt_probs(xₜ, yₜ, heightₜ))
    origin_to_object = norm_3d(xₜ, yₜ, heightₜ)
    rₜ ~ Cat(truncated_discretized_guassian(origin_to_object, 1.0, Rs()), Rs())
    exact_θ ~ LCat(θs())(truncated_discretized_gaussian(acos(x / (exact_r * cos(ϕ))), θstep(), θs()))
    exact_ϕ ~ LCat(ϕs())(truncated_discretized_gaussian(asin(h / exact_r), ϕstep(), ϕs()))

    return (moving_in_depthₜ, vₜ, heightₜ, xₜ, yₜ, rₜ)
end

norm_3d(x, y, z) = sqrt(x^2 + y^2 + z^2)

# TODO: possible improvements:
# 1. maybe just make this a lookup table?
# 2. first round to low-res, then have cheap approximate lookup table
# @gen (static) function norm_3d(x, y, z)
#     xy = x^2 + y^2
#     xzy = xy + z^2
#     return sqrt(xzy)
# end

# θ is azimuth
@gen (static) function obs_model(moving_in_depth, v, height, x, y, r)
    exact_r = norm_3d(x, y, height)
    exact_ϕ = asin(height / exact_r)
    exact_θ = acos(x / (exact_r * cos(exact_ϕ)))

    ϕ ~ Cat(truncated_discretized_gaussian(exact_ϕ, 0.4, ϕs()))
    θ ~ Cat(truncated_discretized_gaussian(exact_θ, 0.4, θs()))

    return (θ, ϕ)
end

@gen (static) function step_proposal(
    moving_in_depthₜ₋₁, vₜ₋₁, heightₜ₋₁, xₜ₋₁, yₜ₋₁, rₜ₋₁, θₜ, ϕₜ # θ and ϕ are noisy
)
    # instead of sampling (x, y, h) then computing r (as we do in the model)
    # in the proposal we sample (r, x, y) and then compute h

    exact_θ ~ LCat(θs())(truncated_discretized_gaussian(θₜ, 0.05, θs()))
    exact_ϕ ~ LCat(ϕs())(truncated_discretized_gaussian(ϕₜ, 0.05, ϕs()))

    rₜ ~ Cat(truncated_discretized_gaussian(rₜ₋₁, 3.0, Rs()))

    # now compute x, y, height (almost deterministically, plus some noise)

    exact_x = rₜ * cos(exact_ϕ) * cos(exact_θ)
    exact_y = rₜ * cos(exact_ϕ) * sin(exact_θ)
    exact_height = r * sin(exact_θ)

    # size in absolute terms is obtained by the az alt divs being discrete 
    # and az alt not having fixed xyz transforms when distant. 

    xₜ ~ LCat(Xs())(truncated_discretized_gaussian(exact_x, 1.0, Xs()))
    yₜ ~ LCat(Ys())(truncated_discretized_gaussian(exact_y, 1.0, Ys()))
    heightₜ ~ LCat(Heights())(truncated_discretized_gaussian(exact_height, 1.0, Heights()))
    
    vₜ ~ LCat(Vels())(maybe_one_off(yₜ - yₜ₋₁, .4, Vels()))
    
    moving_in_depthₜ ~ bernoulli(moving_in_depthₜ₋₁ ? 1.0 : 0.0)

    return (moving_in_depthₜ, vₜ, heightₜ, xₜ, yₜ, rₜ)
end

@gen (static) function initial_proposal(θₜ, ϕₜ)
end