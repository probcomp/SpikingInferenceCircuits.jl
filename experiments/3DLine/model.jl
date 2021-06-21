using Gen
using Distributions
include("../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

include("../../src/ProbEstimates/ProbEstimates.jl")
ProbEstimates.use_perfect_weights!()

include("model_utils.jl")
include("model_hyperparams.jl")

Xs() = 1:10
Ys() = -5:5
Heights() = 1:10
Rs() = 0:Int(ceil(norm_3d(Xs[end], Ys[end], Heights[end])))

@gen (static) function initial_model()
    x = 5
    y = 0
    height = 5
    moving_in_depth = true
    r = Int(round(#= TODO =#))
    
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
    xₜ ~ Cat(moving_in_depthₜ ? maybe_one_off(xₜ₋₁ + v,  0.2, Xs()): onehot(xₜ₋₁, Xs())) # TODO: fill in
    yₜ ~ Cat(maybe_one_off(yₜ₋₁ + 0.2, Ys()))

    origin_to_object ~ norm_3d(xₜ, yₜ, heightₜ)
    rₜ ~ Cat(onehot(origin_to_object), Rs())
    exact_θ ~ Cat(onehot(acos(x / (exact_r * cos(ϕ))), θs()))
    exact_ϕ ~ Cat(onehot(asin(h / exact_r), ϕs()))

    # rₜ ~ LCat(Rs())(truncated_discretized_gaussian(
    #     origin_to_object, 2.0, Rs()
    # ))

    return (moving_in_depthₜ, vₜ, heightₜ, xₜ, yₜ, rₜ)
end

# TODO: possible improvements:
# 1. maybe just make this a lookup table?
# 2. first round to low-res, then have cheap approximate lookup table
@gen (static) function norm_3d(x, y, z)
    xy = x^2 + y^2
    xzy = xy + z^2
    return sqrt(xzy)
end

# θ is azimuth
@gen (static) function obs_model(moving_in_depth, v, height, x, y, r)
    exact_r ~ norm_3d(x, y, height)
    exact_ϕ = asin(h / exact_r)
    exact_θ = acos(x / (exact_r * cos(ϕ)))

    ϕ ~ Cat(trunacated_discretized_gaussian(exact_ϕ, 0.4, ϕs()))
    θ ~ Cat(trunacated_discretized_gaussian(exact_θ, 0.4, θs()))

    return (θ, ϕ)
end

model = @DynamicModel(initial_model, step_model, obs_model, Num_Latent_Vars)

@gen (static) function step_proposal(
    moving_in_depthₜ₋₁, vₜ₋₁, heightₜ₋₁, xₜ₋₁, yₜ₋₁, rₜ₋₁, θₜ, ϕₜ # θ and ϕ are noisy
)
    # instead of sampling (x, y, h) then computing r (as we do in the model)
    # in the proposal we sample (r, x, y) and then compute h

    # we have the prevoius x,y,h, r and current angles
    #  
    
    exact_θ ~ LCat(θs())(truncated_discretized_gaussian(θₜ, 0.05, θs()))
    exact_ϕ ~ LCat(ϕs())(truncated_discretized_gaussian(ϕₜ, 0.05, ϕs()))

    rₜ ~ Cat(truncated_discretized_gaussian(rₜ₋₁, 3.0, Rs()))

    # now deterministically compute x, y, height

    exact_x = rₜ * cos(exact_ϕ) * cos(exact_θ)
    exact_y = rₜ * cos(exact_ϕ) * sin(exact_θ)
    exact_height = r * sin(exact_θ)

    xₜ ~ LCat(Xs())(onehot(exact_x, Xs()))
    yₜ ~ LCat(Ys())(onehot(exact_y, Ys()))
    heightₜ ~ LCat(Heights())(onehot(exact_height, Heights()))
    
    # xₜ ~ Cat(truncated_discretized_gaussian(rₜ*cos(ϕ)*cos(θ), 1.5, Xs()))
    # yₜ ~ Cat(truncated_discretized_gaussian(rₜ*cos(ϕ)*sin(θ), 1.5, Ys()))

    # ϕ_exact = inverse_cosine()
    # (rₜ, xₜ, yₜ)
    # heightₜ ~ Cat(onehot(rₜ*sin(ϕ_exact), Heights()))
    
    vₜ ~ LCat(Vs(), maybe_one_off(yₜ - yₜ₋₁, .4, Vs()))
    
    moving_in_depthₜ ~ bernoulli(moving_in_depthₜ₋₁ ? 1.0 : 0.0)

    return (moving_in_depthₜ, vₜ, heightₜ, xₜ, yₜ, rₜ)
end

@gen (static) function initial_proposal(θₜ, ϕₜ)
end