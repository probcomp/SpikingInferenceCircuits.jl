using Gen
using Distributions
using Colors
using GLMakie
using StatsBase
using GeometryBasics
using FileIO
import NaNMath as nm

neg_to_inf(x) = x <= 0 ? Inf : x
norm_3d(x, y, z) = sqrt(x^2 + y^2 + z^2)
round_to_pt1(x) = round(x, digits=1)

include("../model_utils.jl")
include("model_hyperparams_med.jl")
include("labeled_categorical.jl")

function max_distance_inside_grid(ϕ, θ)
    max_x_boundary = Xs()[end] / (cos(ϕ) * cos(θ))
    max_y_pos_boundary = neg_to_inf(Ys()[end] / (cos(ϕ) * sin(θ)))
    max_y_neg_boundary = neg_to_inf(Ys()[1] / (cos(ϕ) * sin(θ)))
    max_z_boundary = Zs()[end] / sin(ϕ)
    r_max = floor(minimum([max_x_boundary, max_y_pos_boundary,
                     max_y_neg_boundary, max_z_boundary]))
    return r_max
end

@gen (static) function initial_model()
    vxₜ = { :vxₜ } ~ LabeledCategorical(Vels())(unif(Vels()))
    vyₜ = { :vyₜ } ~ LabeledCategorical(Vels())(unif(Vels()))
    vzₜ = { :vzₜ } ~ LabeledCategorical(Vels())(unif(Vels()))
    xₜ = { :xₜ } ~ categorical(unif(Xs()))
    yₜ = { :yₜ } ~ LabeledCategorical(Ys())(unif(Ys()))
    zₜ = { :zₜ } ~ categorical(unif(Zs()))
end

@gen (static) function step_model(vxₜ₋₁, vyₜ₋₁, vzₜ₋₁, xₜ₋₁, yₜ₋₁, zₜ₋₁)
    vxₜ = { :vxₜ } ~ LabeledCategorical(Vels())(maybe_one_or_two_off(vxₜ₋₁, 0.2, Vels()))
    vyₜ = { :vyₜ } ~ LabeledCategorical(Vels())(maybe_one_or_two_off(vyₜ₋₁, 0.2, Vels()))
    vzₜ = { :vzₜ } ~ LabeledCategorical(Vels())(maybe_one_or_two_off(vzₜ₋₁, 0.2, Vels()))
    xₜ = { :xₜ } ~ categorical(maybe_one_off(xₜ₋₁ + vxₜ, .2, Xs()))
    yₜ = { :yₜ } ~ LabeledCategorical(Ys())(maybe_one_off(yₜ₋₁ + vyₜ, .2, Ys()))
    zₜ = { :zₜ } ~ categorical(maybe_one_off(zₜ₋₁ + vzₜ, .2, Zs()))
end

@gen (static) function transient_state_model(xₜ, yₜ, zₜ)
    true_r = round(norm_3d(xₜ, yₜ, zₜ))
    true_ϕ = { :true_ϕ } ~ LabeledCategorical(ϕs())(truncated_discretized_gaussian(
        round_to_pt1(nm.asin(zₜ / true_r)), .1, ϕs()))
    true_θ = { :true_θ } ~ LabeledCategorical(θs())(truncated_discretized_gaussian(
        round_to_pt1(nm.atan(yₜ / xₜ)), .1, θs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    r_probvec = normalize(
        vcat(maybe_one_or_two_off(
            true_r <= r_max ? true_r : r_max, .2, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :rₜ } ~ LabeledCategorical(Rs())(r_probvec)
end

@gen (static) function obs_model(true_ϕ, true_θ)
    obs_ϕ = { :obs_ϕ } ~ LabeledCategorical(ϕs())(truncated_discretized_gaussian(true_ϕ, 0.1, ϕs()))
    obs_θ = { :obs_θ } ~ LabeledCategorical(θs())(truncated_discretized_gaussian(true_θ, 0.1, θs()))
    return (obs_θ, obs_ϕ)
end

@load_generated_functions()