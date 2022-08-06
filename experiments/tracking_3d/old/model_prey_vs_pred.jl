using Gen
using Distributions
using Colors
using GLMakie
using StatsBase
using GeometryBasics
using FileIO
import NaNMath as nm

# if you have time, add a second type of animal. this one goes -1, 0, or 1 in every direction. 

# i think the bug here is that the model accounts for all distances, while
# the proposal is bound by the grid. what can happen is that at the edges, the
# model will assign probabilities to impossible distances. also make the model bound by the grid whenplacre
# choosing distances!

# pseudo-marginal tumbling state (draw uniform, draw from previous).
# 1D trajectories in paramecia.

# note there are two source of uncertainty at play. if its moving downwards, it can be moving in depth or height. so there are automatically two different explanations for the same phenomenon. it will get confused this way. number 2, the different velocities of the two stimulus types can be a confound. currently i think the model is close to what i want. i would rather velocity be perceived in spherical space. then translated to XYZ space. 



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

# x = back and forth
# y = left and right
# z = up and down (held constant in this model)


@gen (static) function initial_model()
    is_prey = { :is_prey } ~ bernoulli(.5)
    vxₜ_dir = { :vxₜ } ~ LCat(Vels())(unif(Vels()))
    vyₜ_dir = { :vyₜ } ~ LCat(Vels())(unif(Vels()))
    vzₜ_dir = { :vzₜ } ~ LCat(Vels())(unif(Vels()))
    vxₜ = vxₜ_dir * (is_prey ? PreyVelScale() : PredatorVelScale())
    vyₜ = vyₜ_dir * (is_prey ? PreyVelScale() : PredatorVelScale())
    vzₜ = vzₜ_dir * (is_prey ? PreyVelScale() : PredatorVelScale())
    xₜ = { :xₜ } ~ Cat(unif(Xs()))
    yₜ = { :yₜ } ~ LCat(Ys())(unif(Ys()))
    zₜ = { :zₜ } ~ Cat(unif(Zs()))
    true_r = round(norm_3d(xₜ, yₜ, zₜ))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(
        round_to_pt1(nm.asin(zₜ / true_r)), 0.1, ϕs()))
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(
        round_to_pt1(nm.atan(yₜ / xₜ)), 0.1, θs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
   r_probvec = normalize(vcat(truncated_discretized_gaussian(
       true_r <= r_max ? true_r : r_max, 2, Rs())[1:Int(r_max)],
                    zeros(length(Rs())-Int(r_max))))
    rₜ = { :rₜ } ~ LCat(Rs())(r_probvec)
    return (vxₜ, vyₜ, vzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ, is_prey)
end


@gen (static) function step_model(vxₜ₋₁, vyₜ₋₁, vzₜ₋₁, xₜ₋₁, yₜ₋₁, zₜ₋₁, rₜ₋₁, ephi, etheta, is_prey)
    vel_σ = .4
    is_prey = { :is_prey } ~ bernoulli(is_prey ? .99 : .01)
    vxₜ_dir = { :vxₜ } ~ LCat(Vels())(truncated_discretized_gaussian(vxₜ₋₁, vel_σ, Vels()))
    vyₜ_dir = { :vyₜ } ~ LCat(Vels())(truncated_discretized_gaussian(vyₜ₋₁, vel_σ, Vels()))
    vzₜ_dir = { :vzₜ } ~ LCat(Vels())(truncated_discretized_gaussian(vzₜ₋₁, vel_σ, Vels()))
    vxₜ = vxₜ_dir * (is_prey ? PreyVelScale() : PredatorVelScale())
    vyₜ = vyₜ_dir * (is_prey ? PreyVelScale() : PredatorVelScale())
    vzₜ = vzₜ_dir * (is_prey ? PreyVelScale() : PredatorVelScale())
    xₜ = { :xₜ } ~ Cat(truncated_discretized_gaussian(xₜ₋₁ + vxₜ, .2, Xs()))
    yₜ = { :yₜ } ~ LCat(Ys())(truncated_discretized_gaussian(yₜ₋₁ + vyₜ, .2, Ys()))
    zₜ = { :zₜ } ~ Cat(truncated_discretized_gaussian(zₜ₋₁ + vzₜ, .2, Zs()))
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
            true_r <= r_max ? true_r : r_max, 2, Rs())[1:Int(r_max)],
             zeros(length(Rs())-Int(r_max))))
    rₜ = { :rₜ } ~ LCat(Rs())(r_probvec)
    # YES you want dtheta and dphi once you've inferred a distance.

    # YOU WANT TRUE THETA WHICH IS A FUNCTION OF YOUR OBSERVATION AND PREVIOUS VELOCITY.
    # KALMAN FILTER.
    
    

    
    return (vxₜ, vyₜ, vzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ, is_prey)
end

# if this is receiving a sample of r, then it could be shorter than x. 

@gen (static) function obs_model(vxₜ, vyₜ, vzₜ, xₜ, yₜ, zₜ, rₜ, true_ϕ, true_θ)
    obs_ϕ = { :obs_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(true_ϕ, 0.1, ϕs()))
    obs_θ = { :obs_θ } ~ LCat(θs())(truncated_discretized_gaussian(true_θ, 0.1, θs()))
    return (obs_θ, obs_ϕ)
end


# here you probably can run into proposing an unrealistic r xyz combination because
# you're directly making sure you don't propose unrealistic steps for x y and z. but should be fine
# since r is one or two off. 

    # likely issue here is its very possible that delta
    # x y and z could be larger than the velocity. this prob creates nans. think about
    # how to address this problem.

    # if velocity is 1, x can only be 2 greater, one greater, or 0 greater than x prev.
    # if velocity is 0, can only be 1 greater, equal, or one less.
    # if velocity is -1, can be two less, one less, or equal to xprev
    # v can only be one off vprev

    # here compare exact vals to t-1 vals
    # if round(exact) - pos t-1 = 1,  x = maybe one off this delta + pos t-1

# difference between the prior and the proposal is that step_model steps forward by generating a similar velocity, then
# calculating the xyz coord, and calculating a distance (i.e. there is no role for distance perception, and no knowledge of the previous distance).
# the step_proposal observes an az and alt, then says "the distance is probabably similar"; using the sampled distance, you
# sample an x, y, and z and a velocity centered on the difference between the last and previous XYZ states. this way the model
# favors explanations with similar velocities but the proposal on similar distances. proposal will ultimately let you propose bio-realistic distance metrics. 


@gen (static) function step_proposal(vxₜ₋₁, vyₜ₋₁, vzₜ₋₁, xₜ₋₁, yₜ₋₁, zₜ₋₁,
                                     rₜ₋₁, true_ϕ, true_θ, is_prey, obs_θ, obs_ϕ) 
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(obs_θ, 0.2, θs()))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(obs_ϕ, 0.2, ϕs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    σᵣ = is_prey ? 2*PreyVelScale() : 2*PredatorVelScale()
    r_probvec = normalize(vcat(truncated_discretized_gaussian(
        rₜ₋₁ <= r_max ? rₜ₋₁ : r_max, σᵣ, Rs())[1:Int(r_max)],
                               zeros(length(Rs())-Int(r_max))))
    rₜ = { :rₜ } ~ LCat(Rs())(r_probvec)
    x_prop = Int(round(rₜ * cos(true_ϕ) * cos(true_θ)))
    y_prop = Int(round(rₜ * cos(true_ϕ) * sin(true_θ)))
    z_prop = Int(round(rₜ * sin(true_ϕ)))
    
    vx_prop = limit_delta_pos(x_prop, xₜ₋₁, is_prey)
    vy_prop = limit_delta_pos(y_prop, yₜ₋₁, is_prey)
    vz_prop = limit_delta_pos(z_prop, zₜ₋₁, is_prey)
    
    xₜ = { :xₜ } ~ LCat(Xs())(truncated_discretized_gaussian(xₜ₋₁ + vx_prop, .1, Xs()))
    yₜ = { :yₜ } ~ LCat(Ys())(truncated_discretized_gaussian(yₜ₋₁ + vy_prop, .1, Ys()))
    zₜ = { :zₜ } ~ LCat(Zs())(truncated_discretized_gaussian(zₜ₋₁ + vz_prop, .1, Zs()))

    is_prey = { :is_prey } ~ bernoulli(vx_prop > PreyVelScale() || vy_prop > PreyVelScale() || vz_prop > PreyVelScale() ? .05 : .5)
    
    vx_index = scale_velocity(vx_prop, is_prey)
    vy_index = scale_velocity(vy_prop, is_prey)
    vz_index = scale_velocity(vz_prop, is_prey)
    
    vxₜ = { :vxₜ } ~ LCat(Vels())(truncated_discretized_gaussian(vx_index, .4, Vels()))
    vyₜ = { :vyₜ } ~ LCat(Vels())(truncated_discretized_gaussian(vy_index, .4, Vels()))
    vzₜ = { :vzₜ } ~ LCat(Vels())(truncated_discretized_gaussian(vz_index, .4, Vels()))
end


@gen (static) function initial_proposal(obs_θ, obs_ϕ)
    true_θ = { :true_θ } ~ LCat(θs())(truncated_discretized_gaussian(obs_θ, 0.3, θs()))
    true_ϕ = { :true_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(obs_ϕ, 0.3, ϕs()))
    r_max = max_distance_inside_grid(true_ϕ, true_θ)
    # THIS IS A BID FROM MOMENT 1. YOU GET NO MORE BIDS. YOU CAN USE A UNIFORM TOO.
#    rₜ = { :rₜ } ~ LCat(Rs())(
 #       truncated_discretized_gaussian(round(norm_3d(X_init, Y_init, Z_init)), 1, Rs()))
    # SWITCH TO THIS IF YOU WANT COMPLETELY UNINFORMED DISTANCE BIDS TO START 
    l = length(Rs())
    r_probvec = normalize(vcat(ones(Int64(r_max)), zeros(Int64(l-r_max))))
    rₜ = { :rₜ } ~ LCat(Rs())(r_probvec)
    is_prey = { :is_prey } ~ bernoulli(.5)
    x_prop = rₜ * cos(true_ϕ) * cos(true_θ)
    y_prop = rₜ * cos(true_ϕ) * sin(true_θ)
    z_prop = rₜ * sin(true_ϕ)
    # size in absolute terms is obtained by the az alt divs being discrete 
    # and az alt not having fixed xyz transforms when distant.
    xₜ = { :xₜ } ~ LCat(Xs())(truncated_discretized_gaussian(round(x_prop), .4, Xs()))
    yₜ = { :yₜ } ~ LCat(Ys())(truncated_discretized_gaussian(round(y_prop), .4, Ys()))
    zₜ = { :zₜ } ~ LCat(Zs())(truncated_discretized_gaussian(round(z_prop), .4, Zs()))
    # add in conditional velocities here. 
    vxₜ_dir = { :vxₜ } ~ LCat(Vels())(unif(Vels()))
    vyₜ_dir = { :vyₜ } ~ LCat(Vels())(unif(Vels()))
    vzₜ_dir = { :vzₜ } ~ LCat(Vels())(unif(Vels()))
end

function max_distance_inside_grid(ϕ, θ)
    max_x_boundary = Xs()[end] / (cos(ϕ) * cos(θ))
    max_y_pos_boundary = neg_to_inf(Ys()[end] / (cos(ϕ) * sin(θ)))
    max_y_neg_boundary = neg_to_inf(Ys()[1] / (cos(ϕ) * sin(θ)))
    max_z_boundary = Zs()[end] / sin(ϕ)
    r_max = floor(minimum([max_x_boundary, max_y_pos_boundary,
                     max_y_neg_boundary, max_z_boundary]))
    return r_max
end



function limit_delta_pos(p_prop, p_prev, is_prey)
    v_scale = is_prey ? PreyVelScale() : PredatorVelScale() / PreyVelScale()
    if (p_prop - p_prev) > (Vels()[end] * v_scale)
        return Vels()[end] * v_scale
    elseif (p_prop - p_prev) < (Vels()[1] * v_scale)
        return Vels()[1] * v_scale
    else
        return Int(round(p_prop - p_prev))
    end
end


function animate_azalt_trajectory(tr)
    gt_obs_choices = get_choices(tr)
    obs_θ = [gt_obs_choices[:steps => step => :obs => :obs_θ => :val] for step in 1:NSTEPS]
    obs_ϕ = [gt_obs_choices[:steps => step => :obs => :obs_ϕ => :val] for step in 1:NSTEPS]
    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1,1])
    hidedecorations!(ax)
    lines!(ax, obs_θ, obs_ϕ, linestyle=:dash, linewidth=4, color=to_colormap(:thermal, length(obs_θ)))
    scatter!(ax, obs_θ, obs_ϕ, markersize=25, color=to_colormap(:thermal, length(obs_θ)), marker=:rect)
    display(fig)
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
        azalt_matrices[1, findfirst(map(x -> x == choices[:init => :latents => :exact_θ => :val], θs())),
                        findfirst(map(x -> x == choices[:init => :latents => :exact_ϕ => :val], ϕs()))] += 1
        for step in 1:NSTEPS
            obs_θ = choices[:steps => step => :latents => :exact_θ => :val]
            obs_ϕ = choices[:steps => step => :latents => :exact_ϕ => :val]
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

    

function heatmap_pf_results(uw_traces, gt::Trace, nsteps)
    
    depth_indexer = [[:steps, i, :latents, :xₜ, :val] for i in 1:nsteps]
    height_indexer = [[:steps, i, :latents, :zₜ, :val] for i in 1:nsteps]
                                  
    gray_cmap = range(colorant"white", stop=colorant"gray32", length=6)
    true_depth = [extract_submap_value(get_choices(gt), depth_indexer[i]) for i in 1:nsteps]
    true_height = [extract_submap_value(get_choices(gt), height_indexer[i]) for i in 1:nsteps]
    depth_matrix = zeros(nsteps, length(0:Xs()[end]) + 1)
    height_matrix = zeros(nsteps, length(0:Zs()[end]) + 1)
    for t in 1:nsteps
        for tr in uw_traces[end]
            depth_matrix[t, Int64(extract_submap_value(get_choices(tr), depth_indexer[t]))] += 1
            height_matrix[t, Int64(extract_submap_value(get_choices(tr), height_indexer[t]))] += 1
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
    ylims!(ax_height, (0.0, Zs()[end]+2))
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

function render_pf_results(uw_traces, gt_trace, n_steps)
    res = 1000
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
    gt_coords = []
    particle_coords = []
    score_colors = []
    for i in 1:n_steps+1
        step_coords = []
        trace_scores = []
        gt_temp = []
        for (tnum, tr) in enumerate(vcat(gt_trace, uw_traces[i]))
            ch = get_choices(tr)
            if i == 1
                x = extract_submap_value(ch, [:init, :latents, :xₜ, :val])
                y = extract_submap_value(ch, [:init, :latents, :yₜ, :val])
                z = extract_submap_value(ch, [:init, :latents, :zₜ, :val])
            else
                x = extract_submap_value(ch, [:steps, i-1, :latents, :xₜ, :val])
                y = extract_submap_value(ch, [:steps, i-1, :latents, :yₜ, :val])
                z = extract_submap_value(ch, [:steps, i-1, :latents, :zₜ, :val])
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
    for i in 1:n_steps
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
