using CSV
using GLMakie
using DataFrames
using FileIO
using HDF5
using ProbEstimates

include("../model_hyperparams.jl")
include("../model.jl")

preycap_record = h5open("prey_coords.h5", "r")
cutoff = 10

# nsteps is a proxy for amount of bouts

function normalize_wrth_coordinates(sampling_rate)
    dv = round(Int, 62.5 / sampling_rate)
    px = read(preycap_record, "x")
    py = read(preycap_record, "y")
    pz = read(preycap_record, "z")
    θ = read(preycap_record, "az")
    ϕ = read(preycap_record, "alt")
    r = read(preycap_record, "dist")
    npx = [(x / tanksize) * length(Xs()) for x in px]
    npy = [(y / (tanksize / 2)) * y_ub for y in py]
    npz = [(z / (tanksize / 2)) * z_ub for z in pz]
    rounded_x = [isfinite(x) ? round(Int, x) : NaN for x in npx]
    rounded_y = [isfinite(y) ? round(Int, y) : NaN for y in npy]
    rounded_z = [isfinite(z) ? round(Int, z) : NaN for z in npz]
    rounded_θ = [isfinite(az) ? round_to_pt1(az) : NaN for az in θ]
    rounded_ϕ = [isfinite(alt) ? round_to_pt1(alt) : NaN for alt in ϕ]
    rounded_r = [isfinite(dist) ? round(Int, Rs()[end] * (dist / tanksize)) : NaN for dist in r]
    bout_times = read(preycap_record, "BoutInds")
    return [rounded_x[1:dv:end], rounded_y[1:dv:end], rounded_z[1:dv:end],
            rounded_θ[1:dv:end], rounded_ϕ[1:dv:end], rounded_r[1:dv:end], bout_times]
end

# use multiple dispatch to make this non-redundant with make_deterministic_trace

function make_trace_from_realprey(sampling_rate)
    x_traj, y_traj, z_traj,
    true_θ, true_ϕ, true_r, bouts = normalize_wrth_coordinates(sampling_rate)
    n_steps = length(x_traj)
    dx_traj = diff(x_traj)
    dy_traj = diff(y_traj)
    dz_traj = diff(z_traj)
# has to start at X Y Z INIT. First d is the diff between Xinit and x_traj[1]
    x_traj_choice = [(:steps => i => :latents => :x => :val, x) for (i, x) in enumerate(x_traj[2:end])]
    y_traj_choice = [(:steps => i => :latents => :y => :val, y) for (i, y) in enumerate(y_traj[2:end])]
    z_traj_choice = [(:steps => i => :latents => :z => :val, z) for (i, z) in enumerate(z_traj[2:end])]
    dx_traj_choice = [(:steps => i => :latents => :dx => :val, dx) for (i, dx) in enumerate(dx_traj)]
    dy_traj_choice = [(:steps => i => :latents => :dy => :val, dy) for (i, dy) in enumerate(dy_traj)]
    dz_traj_choice = [(:steps => i => :latents => :dz => :val, dz) for (i, dz) in enumerate(dz_traj)]
    # Think deeply about the right answer here for true_r and rt-1. 
    true_ϕ_choice = [(:steps => i => :latents => :true_ϕ => :val, ϕ) for (i, ϕ) in enumerate(true_ϕ[2:end])]    
    true_θ_choice = [(:steps => i => :latents => :true_θ => :val, θ) for (i, θ) in enumerate(true_θ[2:end])]
    r_choice = [(:steps => i => :latents => :r => :val, r) for (i, r) in enumerate(true_r[2:end])]
    obsθ_choice = [(:steps => i => :obs => :obs_θ => :val, θ) for (i, θ) in enumerate(true_θ[2:end])]
    obsϕ_choice = [(:steps => i => :obs => :obs_ϕ => :val, ϕ) for (i, ϕ) in enumerate(true_ϕ[2:end])]
    obsθ_init = (:init => :obs => :obs_θ => :val, true_θ[1])
    obsϕ_init = (:init => :obs => :obs_ϕ => :val, true_ϕ[1])
    x_init = (:init => :latents => :x => :val, X_init)
    y_init = (:init => :latents => :y => :val, Y_init)
    z_init = (:init => :latents => :z => :val, Z_init)
    θ_init = (:init => :latents => :true_θ => :val, true_θ[1])
    ϕ_init = (:init => :latents => :true_ϕ => :val, true_ϕ[1])
    r_init = (:init => :latents => :r => :val, true_r[1])
    tr_choicemap = choicemap(x_init, y_init, z_init, obsθ_init, obsϕ_init, θ_init, ϕ_init,
                             x_traj_choice..., y_traj_choice..., z_traj_choice...,
                             dx_traj_choice..., dy_traj_choice..., dz_traj_choice...,
                             true_θ_choice..., true_ϕ_choice..., 
                             r_choice..., obsϕ_choice..., obsθ_choice...)
    return tr_choicemap
end

# NEXT STEPS:
# filter for long (~150-200 ms) periods where the fish is observing.
# add 9 frames for bout duration then require 10 frames before next bout (19 total).
# then take the most recent 10. 



    

n = normalize_wrth_coordinates()


function plot_3D_trajectory(p_id, n_steps)
    fish_mesh = FileIO.load("zebrafish.obj")
    tanksize = 1888
    xgrid = Xs()
    p3Dcoords = collect(zip(read(preycap_record, "x"),
                            read(preycap_record, "y"),
                            read(preycap_record, "z")))
    coord_iter = Int(round(length(p3Dcoords) / n_steps))
    fig = Figure(resolution=(1000, 1000))
    time_node = Node(1)
    lim = (-1,tanksize, -1, tanksize, -1, tanksize)
    ax3d = Axis3(fig[1,1], limits=lim)
#                  viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim)
    # ax3d = Axis3(fig[1,1],
    #              viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim)
    f_pcoord(t) = p3Dcoords[t]
 #   scatter!(ax3d, lift(x -> f_pcoord(x), time_node), color=:black, markersize=5000)
    meshscatter!(ax3d, [(0, 0, 0)], marker=fish_mesh, color=:lightgray, rotations=Vec3f0(0, 1, 0), markersize=20)
    display(fig)
    for i in 1:length(p3Dcoords)
        time_node[] = i
        sleep(.016)
    end
end


function para_3Dtrajectory_in_modelspace(p_id, n_steps)
    fish_mesh = FileIO.load("zebrafish.obj")
    tanksize = 1888
    xgrid = Xs()
    p3Dcoords = collect(zip(read(preycap_record, "x"),
                            read(preycap_record, "y"),
                            read(preycap_record, "z"))) 
    coord_iter = Int(round(length(p3Dcoords) / n_steps))
 #   scatter!(ax3d, lift(x -> f_pcoord(x), time_node), color=:black, markersize=5000)
    normalized_x = [(x / tanksize)*length(Xs()) + Xs()[1] for x in read(preycap_record, "x")][1:coord_iter:end]
    normalized_y = [(y / tanksize)*length(Ys()) + Ys()[1] for y in read(preycap_record, "y")][1:coord_iter:end]
    normalized_z = [(z / tanksize)*length(Zs()) + Zs()[1] for z in read(preycap_record, "z")][1:coord_iter:end]
    p3gridcoords = collect(zip(normalized_x, normalized_y, normalized_z))
    f_gridcoords(t) = p3gridcoords[t]
    fig = Figure(resolution=(1000, 1000))
    time_node = Node(1)
    lim = (Xs()[1]-10,Xs()[end], Ys()[1], Ys()[end], Zs()[1]-5, Zs()[end])
    ax3d = Axis3(fig[1,1], limits=lim)
    scatter!(ax3d, lift(i -> f_gridcoords(i), time_node), color=:black, markersize=500)
    meshscatter!(ax3d, [(0, 0, 0)], marker=fish_mesh,
                 color=:lightgray, rotations=Vec3f0(0, 1, 0), markersize=.5)
    display(fig)
    for i in 1:length(p3Dcoords)
        time_node[] = i
        sleep(.1)
    end
#                  viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim)
    # ax3d = Axis3(fig[1,1],
    #              viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim)
    
end


function plot_horizontal_trajectory(p_id, start_ind=0, end_ind=1)
    p_df = get_para(p_id)
    if end_ind == 1
        end_ind = length(p_df.x)
    end
    tanksize = 1888
    decimate_by = 5
    xgrid = Xs()
    p_xcoords = p_df.x[cutoff+start_ind:decimate_by:end_ind-cutoff]
    p_zcoords = p_df.z[cutoff+start_ind:decimate_by:end_ind-cutoff]
    p_xz = collect(zip(p_xcoords, p_zcoords))
    fig = Figure(resolution=(1000, 1000))
    time_node = Node(1)
    ax_anim = Axis(fig[1,1], title="Paramecium Trajectory")
    ax_hist = Axis(fig[2, 1], title="Delta X Distribution")
    ylims!(ax_anim, (-.1, .1))
    xlims!(ax_anim, (0, 1888))
    f_pcoord(t, coords) = (coords[t], 0, 0)
    scatter!(ax_anim, lift(x -> f_pcoord(x, p_zcoords), time_node), color=:darkgreen, markersize=20)
    xlims!(ax_anim, (minimum(p_xcoords)-50, maximum(p_xcoords)+50))
    hist!(ax_hist, diff(p_xcoords))
    display(fig)
    for i in 1:length(p_xcoords)
        time_node[] = i
        sleep(.016*decimate_by*5)
    end
    return p_df.x[cutoff:end-cutoff]
end





# can access this with an index, which grabs a row. e.g. f[1].x is the x coord in row 1 
# p1_only.x, y, z will yield the coordinates. 
