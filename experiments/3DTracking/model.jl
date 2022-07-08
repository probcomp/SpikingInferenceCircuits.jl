using Gen
using Distributions
using Colors
using GLMakie
using CairoMakie
using StatsBase
using GeometryBasics
using FileIO
import NaNMath as nm


CairoMakie.activate!(type = "pdf")

# if you have time, add a second type of animal. this one goes -1, 0, or 1 in every direction. 

# pseudo-marginal tumbling state (draw uniform, draw from previous).
# 1D trajectories in paramecia.


# Notes: might be a confound that initial_model draws a distance.


# USE VARIABLES AS BINS 



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

# SUMMARY -- Initial model draws a random Cartesian velocity and position for the point. It samples a distance based on the XYZ position (e.g. current self position is known and a vector to the sampled XYZ position is used as the mean of a gaussian sample for distance. The location of the retinal az and alt coordinate is sampled using trig. This is a proxy of what happens in the brain (prob accomplished by a lookup table). 
#
# scoring under this model should be uniform. initial proposal will make an azimuth altitude calculation, back propose an XYZ value based on a randomly sampled distance. under this model, any choice should be scored equally. cycle works by first seeing something, then backproposing to XYZ space -- without a distance cue, any XYZ position / distance combination is equally likely along the observed az alt vector.

# step proposal calculates a new az alt based on previous retinal velocity and retinal position, and proposes a new distance that is relatively close to the previous distance (i.e. it could come backwards or forwards along the depth vector, but not by a lot). the step model contains information about how things can move in XYZ space independently of the egocentric reference frame. this is where knowledge can get pretty deep if you want (i.e. base the vx, vy, vz on switching models, create different items with different velocity profiles / different markov probabilities for stepping velocity).

# the depth bid is picked up in the tectum / midbrain. cortex will move the XYZ position forward in space based on how it thinks things move and what it knows about where other things are. retina is always tracking az alt (and its always working! think of occlusion where an object passes in front of a barrier...in this case, az alt predictions will be completely perfect but uncertainty changes at the level of the XYZ model, and particles that are more distant than the barrier should immediatley collapse!). in this sense, retina is doing az alt position and velocity estimates, tectum / midbrain is representing those and assigning a depth -- this depth is random if there's no further information. If you have a single dot on a screen moving diagonally, cortex will score XYZ interpretations of persistent depth and perspective traversal EQUALLY. Tectum will assign multiple distances which will be ambiguous to the model. 

@gen (static) function initial_model()
    a = println("in initial model")
    dxₜ = { :dx } ~ LCat(Vels())(unif(Vels()))
    dyₜ = { :dy } ~ LCat(Vels())(unif(Vels()))
    dzₜ = { :dz } ~ LCat(Vels())(unif(Vels()))
    xₜ = { :x } ~ Cat(unif(Xs()))
    yₜ = { :y } ~ LCat(Ys())(unif(Ys()))
    zₜ = { :z } ~ LCat(Zs())(unif(Zs()))
    true_r = round(norm_3d(xₜ, yₜ, zₜ))
    true_rₜ₋₁ = round(norm_3d((xₜ-dxₜ), (yₜ-dyₜ), (zₜ-dzₜ)))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(
        round_to_pt1(nm.asin(zₜ / true_r)), 0.1, ϕs()))
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(
        round_to_pt1(nm.atan(yₜ / xₜ)), 0.1, θs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    r_probvec = normalize(
        vcat(truncated_discretized_gaussian(
            true_r <= r_max ? true_r : r_max, 2, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :r } ~ LCat(Rs())(r_probvec)
    ϕₜ₋₁ = round_to_pt1(nm.asin((zₜ - dzₜ)/ true_rₜ₋₁))
    θₜ₋₁ = round_to_pt1(nm.atan((yₜ - dyₜ) / (xₜ - dxₜ)))
    dϕ = { :dϕ } ~ LCat(SphericalVels())(truncated_discretized_gaussian(true_ϕ - ϕₜ₋₁, .2, SphericalVels()))
    dθ = { :dθ } ~ LCat(SphericalVels())(truncated_discretized_gaussian(true_θ - θₜ₋₁, .2, SphericalVels()))
    return (dxₜ, dyₜ, dzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ, dϕ, dθ)
end

# x = back and forth
# y = left and right
# z = up and down (held constant in this model)
@gen (static) function step_model(dxₜ₋₁, dyₜ₋₁, dzₜ₋₁, xₜ₋₁, yₜ₋₁, zₜ₋₁, rₜ₋₁, true_ϕₜ₋₁, true_θₜ₋₁, dϕₜ₋₁, dθₜ₋₁)
    a = println("in step model")
    dxₜ = { :dx } ~ LCat(Vels())(truncated_discretized_gaussian(dxₜ₋₁, 0.4, Vels()))
    dyₜ = { :dy } ~ LCat(Vels())(truncated_discretized_gaussian(dyₜ₋₁, 0.4, Vels()))
    dzₜ = { :dz } ~ LCat(Vels())(truncated_discretized_gaussian(dzₜ₋₁, 0.4, Vels()))
    xₜ = { :x } ~ Cat(truncated_discretized_gaussian(xₜ₋₁ + dxₜ, .2, Xs()))
    yₜ = { :y } ~ LCat(Ys())(truncated_discretized_gaussian(yₜ₋₁ + dyₜ, .2, Ys()))
    zₜ = { :z } ~ LCat(Zs())(truncated_discretized_gaussian(zₜ₋₁ + dzₜ, .2, Zs()))
    # Here: a stochastic mapping from (x, y, h) -> (r, θ, ϕ)
    # For now: just use dimension-wise discretized Gaussians.
    true_r = round(norm_3d(xₜ, yₜ, zₜ))
    # BUG HERE -- DISTANCE IS SAMPLED, SO R CAN END UP BEING SMALLER THAN THE RESPECTIVE COMPONENTS OF ITS VECTOR.
    # have to tighten up the variance on the truncated gaussian sample for r. 
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(
        round_to_pt1(nm.asin(zₜ / true_r)), .1, ϕs()))
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(
        round_to_pt1(nm.atan(yₜ / xₜ)), .1, θs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    r_probvec = normalize(
        vcat(truncated_discretized_gaussian(
            true_r <= r_max ? true_r : r_max, .4, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :r } ~ LCat(Rs())(r_probvec)
    dϕ = { :dϕ } ~ LCat(SphericalVels())(truncated_discretized_gaussian(round_to_pt1(true_ϕ -  true_ϕₜ₋₁), .1, SphericalVels()))
    dθ = { :dθ } ~ LCat(SphericalVels())(truncated_discretized_gaussian(round_to_pt1(true_θ -  true_θₜ₋₁), .1, SphericalVels()))
    return (dxₜ, dyₜ, dzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ, dϕ, dθ)
end

# if this is receiving a sample of r, then it could be shorter than x. 

@gen (static) function obs_model(dxₜ, dyₜ, dzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ, dϕ, dθ)
    a = println("phis")
    a = println(true_ϕ)
    a = println(dϕ)
    obs_ϕ = { :obs_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(round_to_pt1(true_ϕ - dϕ), 0.1, ϕs()))
    a = println(obs_ϕ)
    obs_θ = { :obs_θ } ~ LCat(θs())(truncated_discretized_gaussian(round_to_pt1(true_θ - dθ), 0.1, θs()))
    return (obs_ϕ, obs_θ)
end

    # if velocity is 1, x can only be 2 greater, one greater, or 0 greater than x prev.
    # if velocity is 0, can only be 1 greater, equal, or one less.
    # if velocity is -1, can be two less, one less, or equal to xprev
    # v can only be one off vprev
    # here compare exact vals to t-1 vals
    # if round(exact) - pos t-1 = 1,  x = maybe one off this delta + pos t-1

# difference between the idea in the prior and the proposal is that step_model steps forward by generating a similar velocity, then
# calculating the xyz coord, and calculating a distance (i.e. there is no role for distance perception, and no knowledge of the previous distance).
# the step_proposal observes an az and alt, then says "the distance is probabably similar"; using the sampled distance, you
# sample an x, y, and z and a velocity centered on the difference between the last and previous XYZ states. this way the model
# favors explanations with similar velocities but the proposal on similar distances. proposal will ultimately let you propose bio-realistic distance metrics. 


@gen (static) function step_proposal(dxₜ₋₁, dyₜ₋₁, dzₜ₋₁, xₜ₋₁, yₜ₋₁, zₜ₋₁,
                                     rₜ₋₁, true_ϕₜ₋₁, true_θₜ₋₁, dϕₜ₋₁, dθₜ₋₁, obs_ϕ, obs_θ) # θ and ϕ are noisy
    # instead of sampling (x, y, h) then computing r (as we do in the model)
    # in the proposal we sample (r, x, y) and then compute h
    
    # martin thinks the current true theta is your previous obs added to your previous velocity and thats that.

    # if you get obs_θ , you can go back and say "real dθ was probably obsθ - (true_θ - dθ : this is what gets you back to the previous assumed theta).
    # i get a new obs_theta. it was generated by true_theta t-1. if its different than true_theta t-1, you likely miscalculated. the real true_theta t-1 is probably obs_theta.
    # so your velocity estimate was probably wrong at t-1; the real velocity was probably obs - (true_theta t-1 - d_theta) (this gets you back to the position before the step), which is likely to be the velocity now.
    a = println("in step proposal")
    dϕ = { :dϕ } ~ LCat(θs())(truncated_discretized_gaussian(round_to_pt1(obs_ϕ - (true_ϕₜ₋₁ - dϕₜ₋₁)), 0.1, ϕs()))
    dθ = { :dθ } ~ LCat(θs())(truncated_discretized_gaussian(round_to_pt1(obs_θ - (true_θₜ₋₁ - dθₜ₋₁)), 0.1, θs()))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(round_to_pt1(obs_ϕ + dϕ), 0.2, ϕs()))
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(round_to_pt1(obs_θ + dθ), 0.2, θs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    r_probvec = normalize(
        vcat(truncated_discretized_gaussian(
            rₜ₋₁ <= r_max ? rₜ₋₁ : r_max, 2, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :r } ~ LCat(Rs())(r_probvec)
    x_prop = rₜ * cos(true_ϕ) * cos(true_θ)
    y_prop = rₜ * cos(true_ϕ) * sin(true_θ)
    z_prop = rₜ * sin(true_ϕ)
    xₜ = { :x } ~ LCat(Xs())(truncated_discretized_gaussian(x_prop, .1, Xs()))
    yₜ = { :y } ~ LCat(Ys())(truncated_discretized_gaussian(y_prop, .1, Ys()))
    zₜ = { :z } ~ LCat(Zs())(truncated_discretized_gaussian(z_prop, .1, Zs()))
    # any choice of v here will be consistent with the model b/c its one or two off in the model.
    dxₜ = { :dx } ~ LCat(Vels())(truncated_discretized_gaussian(x_prop-xₜ₋₁, .4, Vels()))
    dyₜ = { :dy } ~ LCat(Vels())(truncated_discretized_gaussian(y_prop-yₜ₋₁, .4, Vels()))
    dzₜ = { :dz } ~ LCat(Vels())(truncated_discretized_gaussian(z_prop-zₜ₋₁, .4, Vels()))
end

@gen (static) function initial_proposal(obs_ϕ, obs_θ)
    a = println("in initial proposal")
    dϕ = { :dϕ } ~ LCat(SphericalVels())(unif(SphericalVels()))
    dθ = { :dθ } ~ LCat(SphericalVels())(unif(SphericalVels()))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(obs_ϕ + dϕ, 0.2, ϕs()))    
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(obs_θ + dθ, 0.2, θs()))
    # Max distance function guarantees that no proposal leaves the grid. 
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    l = length(Rs())
    r_probvec = normalize(vcat(ones(Int64(r_max)), zeros(Int64(l-r_max))))
  #  rₜ = { :rₜ } ~ LCat(Rs())(r_probvec)
    rₜ = { :r } ~ LCat(Rs())(
        truncated_discretized_gaussian(round(norm_3d(X_init, Y_init, Z_init)),
                                       1, Rs()))
    x_prop = rₜ * cos(true_ϕ) * cos(true_θ)
    y_prop = rₜ * cos(true_ϕ) * sin(true_θ)
    z_prop = rₜ * sin(true_ϕ)
    prev_x_prop = rₜ * cos(obs_ϕ) * cos(obs_θ)
    prev_y_prop = rₜ * cos(obs_ϕ) * sin(obs_θ)
    prev_z_prop = rₜ * sin(obs_ϕ)
    # size in absolute terms is obtained by the az alt divs being discrete 
    # and az alt not having fixed xyz transforms when distant.
    xₜ = { :x } ~ LCat(Xs())(truncated_discretized_gaussian(round(x_prop), 1, Xs()))
    yₜ = { :y } ~ LCat(Ys())(truncated_discretized_gaussian(round(y_prop), 1, Ys()))
    zₜ = { :z } ~ LCat(Zs())(truncated_discretized_gaussian(round(z_prop), 1, Zs()))
    dxₜ = { :dx } ~ LCat(Vels())(truncated_discretized_gaussian(round_to_pt1(x_prop - prev_x_prop), 1, Vels()))
    dyₜ = { :dy } ~ LCat(Vels())(truncated_discretized_gaussian(round_to_pt1(y_prop - prev_y_prop), 1, Vels()))
    dzₜ = { :dz } ~ LCat(Vels())(truncated_discretized_gaussian(round_to_pt1(z_prop - prev_z_prop), 1, Vels()))    
end

function max_distance_inside_grid(ϕ, θ)
    for (i, r) in enumerate(Rs())
        x_prop = r * cos(ϕ) * cos(θ)
        y_prop = r * cos(ϕ) * sin(θ)
        z_prop = r * sin(ϕ)
        if abs(x_prop) > Xs()[end] || abs(y_prop) > Ys()[end] || abs(z_prop) > Zs()[end]
            println(i)
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

# VISUALIZATION FUNCTIONS SHOULD BE MOVED TO A viz.jl FILE IN THIS REPO. 



function render_azalt_trajectory(tr)
    CairoMakie.activate!()
    gt_obs_choices = get_choices(tr)
    obs_θ = [gt_obs_choices[:steps => step => :obs => :obs_θ => :val] for step in 1:NSTEPS]
    obs_ϕ = [gt_obs_choices[:steps => step => :obs => :obs_ϕ => :val] for step in 1:NSTEPS]
    theme = Attributes(Axis = (xminorticksvisible=true, yminorticksvisible=true,
                               xminorgridvisible=true, yminorgridvisible=true))
    fig = with_theme(theme) do
        fig = Figure(resolution=(1000, 1000))
        axs = Axis(fig[1,1],
                   xminorticks=IntervalsBetween(5), yminorticks=IntervalsBetween(5))
        limits!(axs, θs()[1], θs()[end], ϕs()[1], ϕs()[end])
        # for some reason have to add an extra .03 here to get it to fit the .1 x .1 box. 
        scatter!(axs, obs_θ .+ .05, obs_ϕ .+ .05, markersize=.13, markerspace=:data,
                 color=to_colormap(:thermal, length(obs_θ)) , marker=:rect)
        fig
    end
    #    lines!(ax, obs_θ, obs_ϕ, linestyle=:dash, linewidth=4, color=to_colormap(:thermal, length(obs_θ)))
    
    display(fig)
    save("traj.pdf", fig)
end


# just have to make the perfect trace here and you'll be set.
# make everything deterministic.

# velocities are the velocities that got you from x t-1 to x.
# x_traj contains the initial observation and initial position.

# TODO -- make sure this function is correct. Write one more function to render the azalt
# using a choicemap generated from every value EXCEPT the obs. then plot the obs. 

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
    true_rₜ₋₁ = [round(norm_3d((x-dx), (y-dy), (z-dz))) for (x, y, z, dx, dy, dz) in zip(x_traj[2:end], y_traj[2:end], z_traj[2:end], dx_traj, dy_traj, dz_traj)]
    true_ϕ = [round_to_pt1(nm.asin(z / r)) for (z, r) in zip(z_traj, true_r)]
    true_θ = [round_to_pt1(nm.atan(y / x)) for (x, y) in zip(x_traj, y_traj)] 
    true_ϕₜ₋₁ = [round_to_pt1(nm.asin((z - dz)/ r)) for (z, dz, r) in zip(z_traj, dz_traj, true_rₜ₋₁)]
    true_θₜ₋₁ = [round_to_pt1(nm.atan((y - dy) / (x - dx))) for (x, y, dx, dy) in zip(x_traj, y_traj, dx_traj, dy_traj)]
    dϕ = [round_to_pt1(tϕ - tϕₜ₋₁) for (tϕ, tϕₜ₋₁) in zip(true_ϕ, true_ϕₜ₋₁)]
    dθ = [round_to_pt1(tθ - tθₜ₋₁) for (tθ, tθₜ₋₁) in zip(true_θ, true_θₜ₋₁)]
    obsθ = [round_to_pt1(t-d) for (t, d) in zip(true_θ, dθ)]
    obsϕ = [round_to_pt1(t-d) for (t, d) in zip(true_ϕ, dϕ)]
    true_θ_choice = [(:steps => i => :latents => :true_θ => :val, θ) for (i, θ) in enumerate(true_θ)]
    true_ϕ_choice = [(:steps => i => :latents => :true_ϕ => :val, ϕ) for (i, ϕ) in enumerate(true_ϕ)]
    r_choice = [(:steps => i => :latents => :r => :val, r) for (i, r) in enumerate(true_r)]
    dθ_choice = [(:steps => i => :latents => :true_θ => :val, θ) for (i, θ) in enumerate(dθ)]
    dϕ_choice = [(:steps => i => :latents => :true_ϕ => :val, ϕ) for (i, ϕ) in enumerate(dϕ)]
    obsθ_choice = [(:steps => i => :obs => :obs_θ => :val, θ) for (i, θ) in enumerate(obsθ[2:end])]
    obsϕ_choice = [(:steps => i => :obs => :obs_ϕ => :val, ϕ) for (i, ϕ) in enumerate(obsϕ[2:end])]
    obsθ_init = (:init => :obs => :obs_ϕ => :val, obsθ[1]) 
    obsϕ_init = (:init => :obs => :obs_ϕ => :val, obsϕ[1])
    x_init = (:init => :latents => :x => :val, X_init)
    y_init = (:init => :latents => :y => :val, Y_init)
    z_init = (:init => :latents => :z => :val, Z_init)
    r_init = (:init => :latents => :r => :val, true_r[1])
    tr_choicemap = choicemap(x_init, y_init, z_init, obsθ_init, obsϕ_init,
                             x_traj_choice..., y_traj_choice..., z_traj_choice...,
                             dx_traj_choice..., dy_traj_choice..., dz_traj_choice...,
                             true_θ_choice..., true_ϕ_choice..., dθ_choice..., dϕ_choice...,
                             r_choice..., obsϕ_choice..., obsθ_choice...)
    return tr_choicemap
end

    

function animate_azalt_heatmap(tr_list, anim_now)
    azalt_matrices = zeros(NSTEPS+1, length(θs()), length(ϕs()))
    obs_matrices = zeros(NSTEPS+1, length(θs()), length(ϕs()))
    gt_obs_choices = get_choices(tr_list[1])
    obs_matrices[1, findfirst(map(x -> x == gt_obs_choices[:init => :obs => :obs_θ => :val], θs())),
                    findfirst(map(x -> x == gt_obs_choices[:init => :obs => :obs_ϕ => :val], ϕs()))] += 1
    for step in 1:NSTEPS
        obs_θ = gt_obs_choices[:steps => step => :obs => :obs_θ => :val]
        obs_ϕ = gt_obs_choices[:steps => step => :obs => :obs_ϕ => :val]
        obs_matrices[step+1,
                     findfirst(map(x -> x == obs_θ, θs())),
                     findfirst(map(x -> x == obs_ϕ, ϕs()))] += 1
    end
    
    for tr in tr_list
        choices = get_choices(tr)
        azalt_matrices[1, findfirst(map(x -> x == choices[:init => :latents => :true_θ => :val], θs())),
                        findfirst(map(x -> x == choices[:init => :latents => :true_ϕ => :val], ϕs()))] += 1
        for step in 1:NSTEPS
            obs_θ = choices[:steps => step => :latents => :true_θ => :val]
            obs_ϕ = choices[:steps => step => :latents => :true_ϕ => :val]
            azalt_matrices[step+1,
                           findfirst(map(x -> x == obs_θ, θs())),
                           findfirst(map(x -> x == obs_ϕ, ϕs()))] += 1
        end
    end

    
    fig = Figure(resolution=(2000,1000))
    obs_ax = fig[1,1] = Axis(fig)
    azalt_ax = fig[1,2] = Axis(fig)
    time = Node(1)
    hm_exact(t) = azalt_matrices[t, :, :]
    hm_obs(t) = obs_matrices[t, :, :]
    heatmap!(obs_ax, θs(), ϕs(), lift(t -> hm_obs(t), time), colormap=:grays)
    heatmap!(azalt_ax, θs(), ϕs(), lift(t -> hm_exact(t), time), colormap=:grays)
    azalt_ax.aspect = DataAspect()
    obs_ax.aspect = DataAspect()
    obs_ax.xlabel = azalt_ax.xlabel = "Azimuth"
    obs_ax.ylabel = azalt_ax.ylabel = "Altitude"
    if anim_now
        display(fig)
        for i in 1:NSTEPS
            time[] = i
            sleep(.2)
        end
    end
    return obs_matrices, azalt_matrices
end


function extract_submap_value(cmap, symlist) 
    lv = first(symlist)
    if lv == :val
        return cmap[:val]
    elseif length(symlist) == 1
        return cmap[lv]
    else
        cmap_sub = get_submap(cmap, lv)
        extract_submap_value(cmap_sub, symlist[2:end])
    end
end



# you have to make a grid that contains az alt positions over time coded as a heat value, with white as nothing. 

function render_static_trajectories(uw_traces, gt::Trace)
    render_azalt_trajectory(gt)
    GLMakie.activate!()
    res = 700
    fig = Figure(resolution=(2*res, 2*res), figure_padding=50)
    lim = (Xs()[1], Xs()[end], Ys()[1], Ys()[end], Zs()[1], Zs()[end])
    # note perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    preyloc_axis = Axis3(fig[2,1], 
                         viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim,
                         elevation = 1.2*pi, azimuth= .7*pi)
    gt_coords = []
    particle_coords = []
    score_colors = []
#    for i in 1:NSTEPS+1
    for i in 1:NSTEPS+1
        step_coords = []
        trace_scores = []
        gt_temp = []
        for (tnum, tr) in enumerate(vcat(gt, uw_traces[i]))
            ch = get_choices(tr)
            if i == 1
                x = extract_submap_value(ch, [:init, :latents, :x, :val])
                y = extract_submap_value(ch, [:init, :latents, :y, :val])
                z = extract_submap_value(ch, [:init, :latents, :z, :val])
            else
                x = extract_submap_value(ch, [:steps, i-1, :latents, :x, :val])
                y = extract_submap_value(ch, [:steps, i-1, :latents, :y, :val])
                z = extract_submap_value(ch, [:steps, i-1, :latents, :z, :val])
            end
            if tnum == 1
                push!(gt_temp, (x, y, z))
            else
                push!(step_coords, (x, y, z))
                push!(trace_scores, get_score(tr))
            end
            
        end
        push!(particle_coords, step_coords)
        push!(score_colors, trace_scores)
        push!(gt_coords, gt_temp)
    end
    fp(t) = convert(Vector{Point3f0}, particle_coords[t])
    fs(t) = convert(Vector{Float64}, map(f -> isfinite(f) ? .1*log(f) : 0, (-1*score_colors[t])))
    f_gt(t) = convert(Vector{Point3f0}, gt_coords[t])
    #    scatter!(anim_axis, lift(t -> fp(t), time_node), color=lift(t -> fs(t), time_node), colormap=:grays, markersize=msize, alpha=.5)
    lines!(map(x -> convert(Vector{Point3f0}, x)[1] , gt_coords), 
           color=to_colormap(:thermal, NSTEPS+1), linewidth=2)


# PC -> EACH INDEX IS THE VALUE OF EACH PARTICLE AT INDEX STEP. 
     for p_index in 1:NPARTICLES
         lines!(map(x -> convert(Point3f0, x[p_index]), particle_coords), 
                color=to_colormap(:ice, NSTEPS+1), linewidth=2)
     end
    
    
    # lines!(particle_anim_axis, lift(t -> fp(t), time_node), color=gray_w_alpha, markersize=msize, alpha=.5)
    # scatter!(particle_anim_axis, lift(t -> f_gt(t), time_node), color=:red, markersize=msize)

    # scatter!(gt_preyloc_axis, lift(t -> f_gt(t), time_node), color=:red, markersize=msize) #, marker='o')
    # meshscatter!(particle_anim_axis, [(1, 0, 2)],
    #              marker=fish_mesh, color=:gray, rotations=Vec3f0(1, 0, 0), markersize=.75)
    display(fig)
    return particle_coords, gt_coords
end    

function heatmap_pf_results(uw_traces, gt::Trace, nsteps)
    depth_indexer = [[:steps, i, :latents, :x, :val] for i in 1:nsteps]
    height_indexer = [[:steps, i, :latents, :z, :val] for i in 1:nsteps]
    gray_cmap = range(colorant"white", stop=colorant"gray32", length=6)
    true_depth = [extract_submap_value(get_choices(gt), depth_indexer[i]) for i in 1:nsteps]
    true_height = [extract_submap_value(get_choices(gt), height_indexer[i]) for i in 1:nsteps]
    depth_matrix = zeros(nsteps, length(0:Xs()[end]) + 1)
    height_matrix = zeros(nsteps, length(Zs()) + 1)
    for t in 1:nsteps
        for tr in uw_traces[end]
            depth_matrix[t, Int64(extract_submap_value(get_choices(tr), depth_indexer[t]))] += 1
            height_matrix[t, findall(x-> x == extract_submap_value(get_choices(tr), height_indexer[t]), Zs())[1]] += 1
        end
    end
    # also plot the true x values
    fig = Figure(resolution=(1200,1200))
    ax_depth = fig[1, 1] = Axis(fig)
    hm_depth = heatmap!(ax_depth, depth_matrix, colormap=gray_cmap)    
    cbar = fig[1, 2] = Colorbar(fig, hm_depth, label="N Particles")
    ax_height = fig[2, 1] = Axis(fig)
    hm_height = heatmap!(ax_height, height_matrix, colormap=gray_cmap)
    cbar2 = fig[2, 2] = Colorbar(fig, hm_height, label="N Particles")
#    scatter!(ax, [o-.5 for o in observations], [t-.5 for t in 1:times], color=:skyblue2, marker=:rect, markersize=30.0)
    scatter!(ax_depth, [t for t in 1:nsteps], [tx for tx in true_depth], color=:orange, markersize=20.0)
    scatter!(ax_height, [t for t in 1:nsteps], [th for th in true_height], color=:orange, markersize=20.0)
    ax_depth.ylabel = "Depth"
    ax_height.ylabel = "Height"
    ax_height.xlabel = "Time"
    xlims!(ax_height, (.5, nsteps))
    xlims!(ax_depth, (.5, nsteps))
    ylims!(ax_height, (Zs()[1], Zs()[end]+2))
    ylims!(ax_depth, (0.0, Xs()[end]+2))
    println("particle scores")
    println([get_score(tr) for tr in uw_traces[end]])
    display(fig)
#    ax_moving_in_depth = fig[3, 1] = Axis(fig)
    # hist!(ax_moving_in_depth,
    #       [extract_submap_value(
    #           get_choices(tr),
    #           [:steps, NSTEPS, :latents, :moving_in_depthₜ]) for tr in uw_traces[end]])
    return fig
end


# here make an Axis3. animate a scatter plot where
# each particle's xyz coordinate is plotted and the score of the particle is reflected in the color.
# also have the ground truth plotted in a different color.

function animate_pf_results(uw_traces, gt_trace)
    GLMakie.activate!()
    res = 700
    msize = 7000
    c2 = colorant"rgba(255, 0, 255, .25)"
    c1 = colorant"rgba(0, 255, 255, .25)"
    gray_w_alpha = colorant"rgba(60, 60, 60, .2)"
    cmap = range(c1, stop=c2, length=10)
#    fish_mesh = FileIO.load("zebrafish.obj")
    fig = Figure(resolution=(2*res, 2*res), figure_padding=50)
    lim = (Xs()[1], Xs()[end], Ys()[1], Ys()[end], Zs()[1], Zs()[end])
    # note perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    gt_preyloc_axis = Axis3(fig[2,1], 
                            viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim,
                            elevation = 1.2*pi, azimuth= .7*pi)
    particle_anim_axis = Axis3(fig[2,2], 
                               viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim,
                               elevation = 1.2*pi, azimuth= .7*pi)
    azalt_axis = fig[1, 1:2] = Axis(outer_padding= 400, fig)
    observation_matrices, azalt_particle_matrices = animate_azalt_heatmap(uw_traces[end], false)
    # scatter takes a list of tuples. want a list of lists of tuples as an f(t) and lift a node to that.
    time_node = Node(1)
    # for i in 1:n_steps+1
    #     step_coords = []
    #     trace_scores = []
    #     gt_temp = []
    #     for (tnum, tr) in enumerate(vcat(gt_trace, uw_traces[i]))
    #         ch = get_choices(tr)
    #         if i == 1
    #             x = extract_submap_value(ch, [:init, :latents, :x, :val])
    #             y = extract_submap_value(ch, [:init, :latents, :y, :val])
    #             z = extract_submap_value(ch, [:init, :latents, :z, :val])
    #         else
    #             x = extract_submap_value(ch, [:steps, i-1, :latents, :x, :val])
    #             y = extract_submap_value(ch, [:steps, i-1, :latents, :y, :val])
    #             z = extract_submap_value(ch, [:steps, i-1, :latents, :z, :val])
    #         end
    #         if tnum == 1
    #             push!(gt_temp, (x, y, z))
    #         else
    #             push!(step_coords, (x, y, z))
    #             push!(trace_scores, get_score(tr))
    #         end
            
    #     end
    #     push!(particle_coords, step_coords)
    #     push!(score_colors, trace_scores)
    #     push!(gt_coords, gt_temp)
    # end
    gt_coords = []
    particle_coords = []
    score_colors = []
    choices_per_particle = [get_choices(tr) for tr in vcat(gt_trace, uw_traces[end])]
    trace_scores = [get_score(tr) for tr in uw_traces[end]]
    for i in 0:NSTEPS
        step_particle_coords = []
        for (particle_num, ch) in enumerate(choices_per_particle)
            if i == 0
                x = extract_submap_value(ch, [:init, :latents, :x, :val])
                y = extract_submap_value(ch, [:init, :latents, :y, :val])
                z = extract_submap_value(ch, [:init, :latents, :z, :val])
                println("past here")
            else
                x = extract_submap_value(ch, [:steps, i, :latents, :x, :val])
                y = extract_submap_value(ch, [:steps, i, :latents, :y, :val])
                z = extract_submap_value(ch, [:steps, i, :latents, :z, :val])
            end
            if particle_num == 1
                push!(gt_coords, (x, y, z)) 
            else
                push!(step_particle_coords, (x, y, z))
            end
        end
        push!(particle_coords, step_particle_coords)        
    end
    fp(t) = convert(Vector{Point3f0}, particle_coords[t])
    fs(t) = convert(Vector{Float64}, map(f -> isfinite(f) ? .1*log(f) : 0, (-1*score_colors[t])))
    f_gt(t) = [convert(Point3f0, gt_coords[t])]
#    scatter!(anim_axis, lift(t -> fp(t), time_node), color=lift(t -> fs(t), time_node), colormap=:grays, markersize=msize, alpha=.5)
    scatter!(particle_anim_axis, lift(t -> fp(t), time_node), color=gray_w_alpha, markersize=msize, alpha=.5)
    scatter!(particle_anim_axis, lift(t -> f_gt(t), time_node), color=:red, markersize=msize)

    scatter!(gt_preyloc_axis, lift(t -> f_gt(t), time_node), color=:red, markersize=msize) #, marker='o')
    # meshscatter!(particle_anim_axis, [(1, 0, 2)],
    #              marker=fish_mesh, color=:gray, rotations=Vec3f0(1, 0, 0), markersize=.75)
    hm_obs(t) = observation_matrices[t, :, :]
    hm_exact(t) = azalt_particle_matrices[t, :, :]
    heatmap!(azalt_axis, θs(), ϕs(), lift(t -> hm_obs(t), time_node), colormap=:grayC)
    heatmap!(azalt_axis, θs(), ϕs(), lift(t -> hm_exact(t), time_node), colormap=:grayC)
    azalt_axis.aspect = DataAspect()
    azalt_axis.xlabel = "Azimuth"
    azalt_axis.ylabel = "Altitude"
    azalt_axis.title = "2D Observations"
    gt_preyloc_axis.title = "Groundtruth 3D Position"
    particle_anim_axis.title = "Inferred 3D Position"
#    azalt_axis.padding = (20, 20, 20, 20)
#    translate_camera(anim_axis)
    display(fig)
    for i in 1:NSTEPS
        sleep(.5)
        time_node[] = i
    end
    return particle_coords, gt_coords
end


function translate_camera(anim_axis)
    hidedecorations!(anim_axis)
    hidespines!(anim_axis)
    cam = cam3d!(anim_axis.scene)
    cam.projectiontype[] = Makie.Orthographic
    cam.upvector[] = Vec3f0(0, 0, 1)
    cam.lookat[] = Vec3f0(50, 0, 0)
    cam.eyeposition[] = Vec3f0(-5000, -5000, 3000)
    update_cam!(anim_axis.scene, cam)
end


# write an animation for the observations.
# render it from the position of the camera.
# render the noisy observations. 

(l::LCat)(args...) = get_retval(simulate(l, args))




# set truncated gaussian to deterministically go to the end.
# obs model could be outside grid you can't see it. 
