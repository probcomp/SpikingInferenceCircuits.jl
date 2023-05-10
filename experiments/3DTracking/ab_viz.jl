using GLMakie
using CairoMakie

function render_azalt_trajectory(tr, savestring::String; do_obs=true)
    CairoMakie.activate!()
    gt_obs_choices = get_choices(tr)

    if do_obs
        obs_θ = [gt_obs_choices[:steps => step => :obs => :obs_θ => :val] for step in 1:NSTEPS]
        obs_ϕ = [gt_obs_choices[:steps => step => :obs => :obs_ϕ => :val] for step in 1:NSTEPS]
    else
        obs_θ = [gt_obs_choices[:steps => step => :latents => :true_θ => :val] for step in 1:NSTEPS]
        obs_ϕ = [gt_obs_choices[:steps => step => :latents => :true_ϕ => :val] for step in 1:NSTEPS]
    end

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
    save(string(savestring, ".pdf"), fig)
end


# just have to make the perfect trace here and you'll be set.
# make everything deterministic.

# velocities are the velocities that got you from x t-1 to x.
# x_traj contains the initial observation and initial position.

# TODO -- make sure this function is correct. Write one more function to render the azalt
# using a choicemap generated from every value EXCEPT the obs. then plot the obs.

function render_obs_from_particles(uw_traces, particles_to_plot::Int; do_obs=false)
    final_step_particles = [get_choices(tr) for tr in uw_traces[1:particles_to_plot]]
    true_angle_choicemaps = [choicemap() for tr in uw_traces[1:particles_to_plot]]
    if do_obs
        cmap_keys = vcat([(:steps => i => :obs => :obs_θ => :val) for i in 1:NSTEPS], [(:steps => i => :obs => :obs_ϕ => :val) for i in 1:NSTEPS])
    else
        cmap_keys = vcat([(:steps => i => :latents => :true_θ => :val) for i in 1:NSTEPS], [(:steps => i => :latents => :true_ϕ => :val) for i in 1:NSTEPS])
    end
    for (particle, cmap) in enumerate(true_angle_choicemaps)
        for cmk in cmap_keys
            cmap[cmk] = final_step_particles[particle][cmk]
        end
    end
    constrained_traces = [generate(model, (NSTEPS,), tcmap)[1] for tcmap in true_angle_choicemaps]
    for (particle, tr) in enumerate(constrained_traces)
        render_azalt_trajectory(tr, string("particle", particle); do_obs)
    end
    return true_angle_choicemaps
end



function animate_azalt_heatmap(tr_list, anim_now, gridded_model)
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
    if gridded_model
        for tr in tr_list
            choices = get_choices(tr)
            azalt_matrices[1, findfirst(map(x -> x == choices[:init => :latents => :ϕθ => :val][2], θs())),
                            findfirst(map(x -> x == choices[:init => :latents => :ϕθ => :val][1], ϕs()))] += 1
            for step in 1:NSTEPS
                obs_θ = choices[:steps => step => :latents => :ϕθ => :val][2]
                obs_ϕ = choices[:steps => step => :latents => :ϕθ => :val][1]
                azalt_matrices[step+1,
                               findfirst(map(x -> x == obs_θ, θs())),
                               findfirst(map(x -> x == obs_ϕ, ϕs()))] += 1
            end
        end
    else
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

function render_static_trajectories(uw_traces, gt::Trace, from_observer::Bool, yz_gridded::Bool)
    render_azalt_trajectory(gt, "traj")
    GLMakie.activate!()
    res = 700
    fig = Figure(resolution=(2*res, 2*res), figure_padding=0)
    lim = (Xs()[1], Xs()[end], Ys()[1], Ys()[end], Zs()[1], Zs()[end])
    # note perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    preyloc_axis = Axis3(fig[1,1], 
                         viewmode=:fit, aspect=(1,1,1), perspectiveness=0, protrusions=0, limits=lim,
                         elevation = .5, azimuth= .5)

    gt_coords = []
    particle_coords = []
    score_colors = []
    choices_per_particle = [get_choices(tr) for tr in vcat(gt, uw_traces)]
    thermal_colormap = to_colormap(:thermal, NSTEPS+1)
    alpha_val = .1f0
    thermal_colormap_w_alpha = [RGBA(c.r, c.g, c.b, alpha_val) for c in thermal_colormap]    
    trace_scores = [get_score(tr) for tr in uw_traces]
    for i in 0:NSTEPS
        step_particle_coords = []
        for (particle_num, ch) in enumerate(choices_per_particle)
            if i == 0
                x = extract_submap_value(ch, [:init, :latents, :x, :val])
                if yz_gridded
                    y, z = extract_submap_value(ch, [:init, :latents, :yz, :val])
                else
                    y = extract_submap_value(ch, [:init, :latents, :y, :val])
                    z = extract_submap_value(ch, [:init, :latents, :z, :val])
                end
            else
                x = extract_submap_value(ch, [:steps, i, :latents, :x, :val])
                if yz_gridded
                    y, z = extract_submap_value(ch, [:steps, i, :latents, :yz, :val])
                else
                    y = extract_submap_value(ch, [:steps, i, :latents, :y, :val])
                    z = extract_submap_value(ch, [:steps, i, :latents, :z, :val])
                end
            end
            if from_observer
                y = -y
            end
            
            if particle_num == 1
                push!(gt_coords, (x, y, z)) 
            else
                push!(step_particle_coords, (x, y, z))
            end
        end
        push!(particle_coords, step_particle_coords)        
    end
    #    scatter!(anim_axis, lift(t -> fp(t), time_node), color=lift(t -> fs(t), time_node), colormap=:grays, markersize=msize, alpha=.5)
#    lines!(convert(Vector{Point3f0}, gt_coords),
 #          color=to_colormap(:ice, NSTEPS+1), linewidth=2)


# PC -> EACH INDEX IS THE VALUE OF EACH PARTICLE AT INDEX STEP. 
    for p_index in 1:length(uw_traces)
        lines!(map(x -> convert(Point3f0, x[p_index]), particle_coords), 
               color=thermal_colormap_w_alpha, linewidth=2, overdraw=true)
    end
    # lines!(particle_anim_axis, lift(t -> fp(t), time_node), color=gray_w_alpha, markersize=msize, alpha=.5)
    # scatter!(particle_anim_axis, lift(t -> f_gt(t), time_node), color=:red, markersize=msize)

    # scatter!(gt_preyloc_axis, lift(t -> f_gt(t), time_node), color=:red, markersize=msize) #, marker='o')
    # meshscatter!(particle_anim_axis, [(1, 0, 2)],
    #              marker=fish_mesh, color=:gray, rotations=Vec3f0(1, 0, 0), markersize=.75)
    translate_camera(preyloc_axis, from_observer)
    display(fig)
    CairoMakie.activate!()
    save("inferred_trajectories.pdf", fig)
    return particle_coords, gt_coords
end


function plot_full_choicemap(tr)
    GLMakie.activate!()
    full_cmap = get_choices(tr)
    all_latent_varbs = [k[1] for k in get_submaps_shallow(get_submap(full_cmap, (:init => :latents)))]
    all_obs = [k[1] for k in get_submaps_shallow(get_submap(full_cmap, (:init => :obs)))]
    latent_scattervals = [[full_cmap[(:steps => s => :latents => v => :val)] for s in 1:NSTEPS] for v in all_latent_varbs]
    obs_scattervals = [[full_cmap[(:steps => s => :obs => v => :val)] for s in 1:NSTEPS] for v in all_obs]
    for (v, scatvals) in zip(all_latent_varbs, latent_scattervals)
        pushfirst!(scatvals, full_cmap[(:init => :latents => v => :val)])
    end
    for (v, scatvals) in zip(all_obs, obs_scattervals)
        pushfirst!(scatvals, full_cmap[(:init => :obs => v => :val)])
    end
    fig = Figure(resolution=(3000, 1000))
    for (varb_id , (varb, scattervals)) in enumerate(zip(vcat(all_latent_varbs, all_obs), vcat(latent_scattervals, obs_scattervals)))
        axis = fig[1, varb_id] = Axis(fig, title=string(varb))
        scatter!(fig[1, varb_id], collect(enumerate(scattervals)), color=1:NSTEPS, colormap=:thermal, markersize=20)
        if varb in [:true_θ, :true_ϕ, :obs_θ, :obs_ϕ]
            ylims!(axis, (-1.4, 1.4))
        end
        if varb in [:dx, :dy, :dz]
            ylims!(axis, (Vels()[1], Vels()[end]))
        end
    end
    display(fig)
    return fig
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

function animate_pf_results(uw_traces, gt_trace, from_observer::Bool, yz_gridded::Bool)
    GLMakie.activate!()
    res = 700
    if from_observer
        msize = 150px
    else
        msize = 700
    end
    c2 = colorant"rgba(255, 0, 255, .25)"
    c1 = colorant"rgba(0, 255, 255, .25)"
    gray_w_alpha = colorant"rgba(60, 60, 60, .1)"
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
    observation_matrices, azalt_particle_matrices = animate_azalt_heatmap(uw_traces, false, yz_gridded)
    # scatter takes a list of tuples. want a list of lists of tuples as an f(t) and lift a node to that.
    time_node = Node(1)
    gt_coords = []
    particle_coords = []
    score_colors = []
    choices_per_particle = [get_choices(tr) for tr in vcat(gt_trace, uw_traces)]
    trace_scores = [get_score(tr) for tr in uw_traces]
    for i in 0:NSTEPS
        step_particle_coords = []
        println("HEY MAN")
        for (particle_num, ch) in enumerate(choices_per_particle)
            if i == 0
                x = extract_submap_value(ch, [:init, :latents, :x, :val])
                if yz_gridded
                    y, z = extract_submap_value(ch, [:init, :latents, :yz, :val])                    
                else
                    y = extract_submap_value(ch, [:init, :latents, :y, :val])
                    z = extract_submap_value(ch, [:init, :latents, :z, :val])
                end
            else
                x = extract_submap_value(ch, [:steps, i, :latents, :x, :val])
                if yz_gridded
                    y, z = extract_submap_value(ch, [:steps, i, :latents, :yz, :val])
                else
                    y = extract_submap_value(ch, [:steps, i, :latents, :y, :val])
                    z = extract_submap_value(ch, [:steps, i, :latents, :z, :val])
                end
            end
            if from_observer
                y = -y
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
 #   if from_observer
    translate_camera(particle_anim_axis, from_observer)
    translate_camera(gt_preyloc_axis, from_observer)
  #  end
    display(fig)
    for i in 1:NSTEPS
        sleep(.5)
        time_node[] = i
    end
    return particle_coords, gt_coords
end


function translate_camera(anim_axis, observer_pov::Bool)
    hidedecorations!(anim_axis)
    hidespines!(anim_axis)
    cam = cam3d!(anim_axis.scene)
    #    cam.projectiontype[] = Makie.Orthographic
    # i think a 45f0 field of view IS orthographic.
    cam.fov[] = 45f0
    # there is far and near clipping. if stuff is disappearing change this value. 
    if observer_pov
#        cam.far[] = 20                
        cam.eyeposition[] = Vec3f0(0, 0, 0)
        cam.lookat[] = Vec3f0(1, 0, 0)
        cam.upvector[] = Vec3f0(0, 0, 1)

    else
        cam.far[] = 200        
        cam.eyeposition[] = Vec3f0(Xs()[1]-46, Ys()[1], Zs()[end] + 20)
        cam.lookat[] = Vec3f0(Xs()[Int(round(length(Xs()) / 2))],
                              Ys()[Int(round(length(Ys())/2))],
                              Zs()[Int(round(length(Zs())/2))])
        cam.upvector[] = Vec3f0(0, 0, 1)
        axis_vertices = [[(Xs()[end], Ys()[1], Zs()[1]), (Xs()[end], Ys()[1], Zs()[end])], 
                         [(Xs()[end], Ys()[1], Zs()[1]), (Xs()[end], Ys()[end], Zs()[1])],
                         [(Xs()[end], Ys()[1], Zs()[1]), (Xs()[1], Ys()[1], Zs()[1])]]
        full_grid_vertices = [[(Xs()[end], Ys()[end], Zs()[end]), (Xs()[end], Ys()[1], Zs()[end])], 
                              [(Xs()[end], Ys()[end], Zs()[end]), (Xs()[end], Ys()[end], Zs()[1])],
                              [(Xs()[1], Ys()[1], Zs()[1]), (Xs()[1], Ys()[end], Zs()[1])], 
                              [(Xs()[end], Ys()[end], Zs()[end]), (Xs()[1], Ys()[end], Zs()[end])],
                              [(Xs()[end], Ys()[end], Zs()[1]), (Xs()[1], Ys()[end], Zs()[1])],
                              [(Xs()[1], Ys()[end], Zs()[end]), (Xs()[1], Ys()[end], Zs()[1])]
                              ]

        GLMakie.scale!(anim_axis.scene, 1, -1, 1)
        for gv in vcat(axis_vertices, full_grid_vertices)
            lines!(anim_axis, gv, color="black", overdraw=true, linewidth=2)
            #linesegments!(anim_axis, gv, color="black")
            continue
        end
        
    end
    update_cam!(anim_axis.scene, cam)
end


# write an animation for the observations.
# render it from the position of the camera.
# render the noisy observations. 

(l::LCat)(args...) = get_retval(simulate(l, args))

