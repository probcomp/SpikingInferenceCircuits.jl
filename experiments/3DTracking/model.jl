using Gen
using Distributions
using Colors
using StatsBase
using GeometryBasics
using FileIO
import NaNMath as nm

# "exact" is another word for "true"

# i think the bug here is that the model accounts for all distances, while
# the proposal is bound by the grid. what can happen is that at the edges, the
# model will assign probabilities to impossible distances. also make the model bound by the grid whenplacre
# choosing distances!

# NOTES 8/3/2021
# the nanmath and the isfinite calls in onehot, truncated_discretized_gaussian. After running again, all scores are NaN. 
# Usually on a 20 particle run get mostly non-nan scores. May be worth debugging a bit with George.
# Re-added nanmath and its fine. It's the isfinite calls in the distributions. 
# like the drawings from Xuan's paper.

# pseudo-marginal tumbling state (draw uniform, draw from previous).
# 1D trajectories in paramecia.

using ProbEstimates: Cat, LCat

include("model_utils.jl")
include("model_hyperparams.jl")

neg_to_inf(x) = x <= 0 ? Inf : x
norm_3d(x, y, z) = sqrt(x^2 + y^2 + z^2)
round_to_pt1(x) = round(x, digits=1)

# x, y, zₜ are the current positions at time t.
# vx vy and vz are the velocities that move the animal from xt-1 to xt
# in the first step, these have no impact b/c the initial position is drawn. 
# helpful to think of v here as VThatLeadtoXYZInit

@gen (static) function initial_model()
    dxₜ = { :dx } ~ LCat(Vels())(unif(Vels()))
    dyₜ = { :dy } ~ LCat(Vels())(unif(Vels()))
    dzₜ = { :dz } ~ LCat(Vels())(unif(Vels()))
    xₜ = { :x } ~ Cat(unif(Xs()))
    yₜ = { :y } ~ LCat(Ys())(unif(Ys()))
    zₜ = { :z } ~ LCat(Zs())(unif(Zs()))
    true_r = round(norm_3d(xₜ, yₜ, zₜ))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(
        round_to_pt1(nm.asin(zₜ / true_r)), 0.1, ϕs()))
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(
        round_to_pt1(nm.atan(yₜ / xₜ)), 0.1, θs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    r_probvec = normalize(
        vcat(truncated_discretized_gaussian(
            true_r <= r_max ? true_r : r_max, .2, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :r } ~ LCat(Rs())(r_probvec)
    return (dxₜ, dyₜ, dzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ)
end

# x = back and forth
# y = left and right
# z = up and down (held constant in this model)
@gen (static) function step_model(dxₜ₋₁, dyₜ₋₁, dzₜ₋₁, xₜ₋₁, yₜ₋₁, zₜ₋₁, rₜ₋₁, true_ϕₜ₋₁, true_θₜ₋₁)
    dxₜ = { :dx } ~ LCat(Vels())(truncated_discretized_gaussian(dxₜ₋₁, 0.2, Vels()))
    dyₜ = { :dy } ~ LCat(Vels())(truncated_discretized_gaussian(dyₜ₋₁, 0.2, Vels()))
    dzₜ = { :dz } ~ LCat(Vels())(truncated_discretized_gaussian(dzₜ₋₁, 0.2, Vels()))
    xₜ = { :x } ~ Cat(truncated_discretized_gaussian(xₜ₋₁ + dxₜ, .2, Xs()))
    yₜ = { :y } ~ LCat(Ys())(truncated_discretized_gaussian(yₜ₋₁ + dyₜ, .2, Ys()))
    zₜ = { :z } ~ LCat(Zs())(truncated_discretized_gaussian(zₜ₋₁ + dzₜ, .2, Zs()))
    # Here: a stochastic mapping from (x, y, h) -> (r, θ, ϕ)
    # For now: just use dimension-wise discretized Gaussians.
    true_r = round(norm_3d(xₜ, yₜ, zₜ))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(
        round_to_pt1(nm.asin(zₜ / true_r)), .1, ϕs()))
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(
        round_to_pt1(nm.atan(yₜ / xₜ)), .1, θs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    r_probvec = normalize(
        vcat(truncated_discretized_gaussian(
            true_r <= r_max ? true_r : r_max, .2, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :r } ~ LCat(Rs())(r_probvec)
    return (dxₜ, dyₜ, dzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ)
end


@gen (static) function obs_model(dxₜ, dyₜ, dzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ)
    # can't propose to these b/c they are the final observations we're scoring.
    # have to propose to the exact theta and phi.
    obs_ϕ = { :obs_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(round_to_pt1(true_ϕ), 0.1, ϕs()))
    obs_θ = { :obs_θ } ~ LCat(θs())(truncated_discretized_gaussian(round_to_pt1(true_θ), 0.1, θs()))
    return (obs_θ, obs_ϕ)
end


@gen (static) function step_proposal(dxₜ₋₁, dyₜ₋₁, dzₜ₋₁, xₜ₋₁, yₜ₋₁, zₜ₋₁,
                                     rₜ₋₁, true_ϕ, true_θ, obs_θ, obs_ϕ) # θ and ϕ are noisy
    # instead of sampling (x, y, h) then computing r (as we do in the model)
    # in the proposal we sample (r, x, y) and then compute h
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(obs_θ, 0.2, θs()))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(obs_ϕ, 0.2, ϕs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    r_probvec = normalize(
        vcat(truncated_discretized_gaussian(
            rₜ₋₁ <= r_max ? rₜ₋₁ : r_max, .6, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :r } ~ LCat(Rs())(r_probvec)
    x_prop = rₜ * cos(true_ϕ) * cos(true_θ)
    y_prop = rₜ * cos(true_ϕ) * sin(true_θ)
    z_prop = rₜ * sin(true_ϕ)

    xₜ = { :x } ~ LCat(Xs())(truncated_discretized_gaussian(x_prop, .1, Xs()))
    yₜ = { :y } ~ LCat(Ys())(truncated_discretized_gaussian(y_prop, .1, Ys()))
    zₜ = { :z } ~ LCat(Zs())(truncated_discretized_gaussian(z_prop, .1, Zs()))
    # any choice of v here will be consistent with the model b/c its one or two off in the model.
    vxₜ = { :dx } ~ LCat(Vels())(truncated_discretized_gaussian(x_prop-xₜ₋₁, .1, Vels()))
    vyₜ = { :dy } ~ LCat(Vels())(truncated_discretized_gaussian(y_prop-yₜ₋₁, .1, Vels()))
    vzₜ = { :dz } ~ LCat(Vels())(truncated_discretized_gaussian(z_prop-zₜ₋₁, .1, Vels()))
end

@gen (static) function initial_proposal(obs_θ, obs_ϕ)
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(obs_θ, 0.2, θs()))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(obs_ϕ, 0.2, ϕs()))
    # r_max on the first draw is guaranteed to not leave the cube
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    l = length(Rs())
 #   r_probvec = normalize(vcat(ones(Int64(r_max)), zeros(Int64(l-r_max))))
#    rₜ = { :rₜ } ~ LCat(Rs())(r_probvec)
    rₜ = { :r } ~ LCat(Rs())(truncated_discretized_gaussian(round(norm_3d(X_init, Y_init, Z_init)),
                                                    .6, Rs()))
    x_prop = rₜ * cos(true_ϕ) * cos(true_θ)
    y_prop = rₜ * cos(true_ϕ) * sin(true_θ)
    z_prop = rₜ * sin(true_ϕ)
    # size in absolute terms is obtained by the az alt divs being discrete 
    # and az alt not having fixed xyz transforms when distant.
    xₜ = { :x } ~ LCat(Xs())(truncated_discretized_gaussian(round(x_prop), .2, Xs()))
    yₜ = { :y } ~ LCat(Ys())(truncated_discretized_gaussian(round(y_prop), .2, Ys()))
    zₜ = { :z } ~ LCat(Zs())(truncated_discretized_gaussian(round(z_prop), .2, Zs()))
    dxₜ = { :dx } ~ LCat(Vels())(unif(Vels()))
    dyₜ = { :dy } ~ LCat(Vels())(unif(Vels()))
    dzₜ = { :dz } ~ LCat(Vels())(unif(Vels()))
end


function max_distance_inside_grid(ϕ, θ)
    for (i, r) in enumerate(Rs())
        x_prop = r * cos(ϕ) * cos(θ)
        y_prop = r * cos(ϕ) * sin(θ)
        z_prop = r * sin(ϕ)
        if abs(x_prop) > Xs()[end] || abs(y_prop) > Ys()[end] || abs(z_prop) > Zs()[end]
            return Rs()[i-1]
        end
    end
end


function limit_delta_pos(p_prop, p_prev)
    if p_prop - p_prev > Vels()[end]
        return Vels()[end]
    elseif p_prop - p_prev < Vels()[1]
        return Vels()[1]
    else
        return p_prop - p_prev
    end
end

function make_deterministic_trace()
    x_traj = vcat([X_init], [X_init for i in 1:NSTEPS])
    y_traj = vcat([Y_init], [Y_init + i for i in 1:NSTEPS])
    z_traj = vcat([Z_init], [Z_init for i in 1:NSTEPS])
# has to start at X Y Z INIT. First d is the diff between Xinit and x_traj[1]
    dx_traj = diff(x_traj)
    dy_traj = diff(y_traj)
    dz_traj = diff(z_traj)
    x_traj_choice = [(:steps => i => :latents => :x => :val, x) for (i, x) in enumerate(x_traj[2:end])]
    y_traj_choice = [(:steps => i => :latents => :y => :val, y) for (i, y) in enumerate(y_traj[2:end])]
    z_traj_choice = [(:steps => i => :latents => :z => :val, z) for (i, z) in enumerate(z_traj[2:end])]
    dx_traj_choice = [(:steps => i => :latents => :dx => :val, dx) for (i, dx) in enumerate(dx_traj)]
    dy_traj_choice = [(:steps => i => :latents => :dy => :val, dy) for (i, dy) in enumerate(dy_traj)]
    dz_traj_choice = [(:steps => i => :latents => :dz => :val, dz) for (i, dz) in enumerate(dz_traj)]
    # Think deeply about the right answer here for true_r and rt-1. 
    true_r = [round(norm_3d(x, y, z)) for (x, y, z) in zip(x_traj, y_traj, z_traj)]
    true_ϕ = [round_to_pt1(nm.asin(z / r)) for (z, r) in zip(z_traj, true_r)]
    true_θ = [round_to_pt1(nm.atan(y / x)) for (x, y) in zip(x_traj, y_traj)] 
    true_θ_choice = [(:steps => i => :latents => :true_θ => :val, θ) for (i, θ) in enumerate(true_θ)]
    true_ϕ_choice = [(:steps => i => :latents => :true_ϕ => :val, ϕ) for (i, ϕ) in enumerate(true_ϕ)]
    r_choice = [(:steps => i => :latents => :r => :val, r) for (i, r) in enumerate(true_r)]
    obsθ_choice = [(:steps => i => :obs => :obs_θ => :val, θ) for (i, θ) in enumerate(true_θ)]
    obsϕ_choice = [(:steps => i => :obs => :obs_ϕ => :val, ϕ) for (i, ϕ) in enumerate(true_ϕ)]
    obsθ_init = (:init => :obs => :obs_ϕ => :val, true_θ[1])
    obsϕ_init = (:init => :obs => :obs_ϕ => :val, true_ϕ[1])
    x_init = (:init => :latents => :x => :val, X_init)
    y_init = (:init => :latents => :y => :val, Y_init)
    z_init = (:init => :latents => :z => :val, Z_init)
    r_init = (:init => :latents => :r => :val, true_r[1])
    tr_choicemap = choicemap(x_init, y_init, z_init, obsθ_init, obsϕ_init,
                             x_traj_choice..., y_traj_choice..., z_traj_choice...,
                             dx_traj_choice..., dy_traj_choice..., dz_traj_choice...,
                             true_θ_choice..., true_ϕ_choice..., 
                             r_choice..., obsϕ_choice..., obsθ_choice...)
    return tr_choicemap
end




    
