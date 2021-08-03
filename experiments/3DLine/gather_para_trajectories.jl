using CSV
using GLMakie
using DataFrames
using FileIO

include("model_hyperparams.jl")

para_file = select!(CSV.File("para_coords.csv") |> DataFrame, Not(:Column1))

get_para(p_id) = filter(:para_id => p -> p == p_id, para_file)


# nsteps is a proxy for amount of bouts

function plot_para_trajectory(p_id, n_steps, align_w_model)

    fish_mesh = FileIO.load("zebrafish.obj")
    p_df = get_para(p_id)
    tanksize = 1888
    xgrid = Xs()
    p3Dcoords = collect(zip(p_df.x, p_df.y, p_df.z))
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
    normalized_x = [(x / tanksize)*length(Xs()) + Xs()[1] for x in p_df.x][1:coord_iter:end]
    normalized_y = [(y / tanksize)*length(Ys()) + Ys()[1] for y in p_df.y][1:coord_iter:end]
    normalized_z = [(z / tanksize)*length(Zs()) + Zs()[1] for z in p_df.z][1:coord_iter:end]
    return collect(zip(normalized_x, normalized_y, normalized_z))
end


function para_trajectory_in_modelspace(p_id, n_steps)
    fish_mesh = FileIO.load("zebrafish.obj")
    p_df = get_para(p_id)
    tanksize = 1888
    xgrid = Xs()
    p3Dcoords = collect(zip(p_df.x, p_df.y, p_df.z))
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







# can access this with an index, which grabs a row. e.g. f[1].x is the x coord in row 1 
# p1_only.x, y, z will yield the coordinates. 
