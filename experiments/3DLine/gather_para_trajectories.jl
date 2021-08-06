using CSV
using GLMakie
using DataFrames
using FileIO

include("model_hyperparams.jl")

para_file = select!(CSV.File("para_coords.csv") |> DataFrame, Not(:Column1))

get_para(p_id) = filter(:para_id => p -> p == p_id, para_file)

cutoff = 10


# nsteps is a proxy for amount of bouts

function plot_3D_trajectory(p_id, n_steps)

    fish_mesh = FileIO.load("zebrafish.obj")
    p_df = get_para(p_id)
    tanksize = 1888
    xgrid = Xs()
    p3Dcoords = collect(zip(p_df.x, p_df.y, p_df.z))[cutoff:end-cutoff]
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
    p_df = get_para(p_id)
    tanksize = 1888
    xgrid = Xs()
    p3Dcoords = collect(zip(p_df.x, p_df.y, p_df.z))[cutoff:end-cutoff]
    coord_iter = Int(round(length(p3Dcoords) / n_steps))
 #   scatter!(ax3d, lift(x -> f_pcoord(x), time_node), color=:black, markersize=5000)
    normalized_x = [(x / tanksize)*length(Xs()) + Xs()[1] for x in p_df.x][1:coord_iter:end]
    normalized_y = [(y / tanksize)*length(Ys()) + Ys()[1] for y in p_df.y][1:coord_iter:end]
    normalized_z = [(z / tanksize)*length(Zs()) + Zs()[1] for z in p_df.z][1:coord_iter:end]
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
