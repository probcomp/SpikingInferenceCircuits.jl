using GLMakie
using Colors

### 2D Visualizations ###

############### Utils ###################
median_idx(pvec) = percentile_idx(pvec, 0.5)
function percentile_idx(pvec, percentile)
    sum_to_here = 0
    for i=1:length(pvec)
        lsum = sum_to_here
        sum_to_here = lsum + pvec[i]

        if sum_to_here > percentile
            fraction = (percentile - lsum) / (sum_to_here - lsum)
            return i - 0.5 + (fraction)
        end
    end

end
mean_idx(pvec) = sum(p * i for (i, p) in enumerate(pvec))

median(varvals)     = pvec      -> median_idx(pvec)        + first(varvals) - 1
mean(varvals)       = pvec      -> mean_idx(pvec)          + first(varvals) - 1
percentile(varvals) = (pvec, p) -> percentile_idx(pvec, p) + first(varvals) - 1
#############################################################################

### Heatmap + groundtruth + obs visualization ###
function plot_variable_over_time!(
    layout, posterior_probvec_at_times;
    times, groundtruth, varname, varvals,
    observations=nothing,
    show_medians=true, show_means=true, show_percentiles=true, show_colorbar=true,
    markersize=15
)
    layout[1, 1] = ax = Axis(layout[1, 1], xlabel="Time", ylabel=varname)
    inf_matrix = hcat(posterior_probvec_at_times...) |> transpose
    maxprob = maximum(inf_matrix)
    hm = heatmap!(ax,
        times, varvals,
        inf_matrix,
        colormap=cgrad([:white, :black], [0., 0.4, 1.0]),
        colorrange = (0, min(maxprob + 0.1, 1.0))
    )
    if show_colorbar
        layout[1, 2] = Colorbar(layout[1, 2], hm, label="Posterior probability")
    end

    if !isnothing(observations)
        obsplt(f) = f(ax,
            [t for t in times],
            [o for o in observations];
            color=:gold,
            markersize
        )
        obs_plts = [obsplt(lines!), obsplt(scatter!)]
    else
        obs_plts = nothing
    end

    # Ground truth pos:
    gtplt(f) = f(ax,
        [t for t in times],
        [tx for tx in groundtruth];
        color=:seagreen,
        markersize
    )
    gt_plts = [gtplt(lines!), gtplt(scatter!)]

    # Medians:
    if show_medians
        medians      = map(median(varvals), posterior_probvec_at_times) 
        medplt(f) = f(ax,
            [t for t in times],
            [med for med in medians];
            color=:navy,
            markersize,
            (f == scatter! ? (marker=:star4,) : ())...
        )
        medplts = [medplt(lines!)] #, medplt(scatter!)]
    else
        medplts = nothing
    end

    # Means:
    if show_means
        means        = map(mean(varvals)  , posterior_probvec_at_times) 
        meanplt(f) = f(ax,
            [t for t in times],
            [m for m in means];
            color=:black,
            (f == scatter! ? (marker=:hline,) : ())...
        )
        meanplts = [meanplt(lines!)]
    else
        meanplts = nothing
    end

    # percentiles:
    if show_percentiles
        percplt(f, p) = f(ax,
            [t for t in times],
            [percentile(varvals)(pvec, p) for pvec in posterior_probvec_at_times],
            color=:crimson
        )
        per5plts = [percplt(lines!, 0.05)]
        per95plts = [percplt(lines!, 0.95)]
    else
        per5plts = nothing
        per95plts = nothing
    end

    return (ax, (obs_plts, gt_plts, medplts, meanplts, per5plts, per95plts))
end

function draw_2d_posterior!(layout, posterior_probability_grids, tr; show_statistics=true, show_colorbars=true)
    vel_layout = GridLayout(); pos_layout = GridLayout()
    layout[1, :] = vel_layout; layout[2, :] = pos_layout
    rowsize!(layout, 1, Relative(1/3))

    times = 0:(get_args(tr)[1])
    pos_observations = [obs_choicemap(tr, t)[:yᵈₜ => :val] for t in times]
    gt_pos           = [latents_choicemap(tr, t)[:xₜ => :val] for t in times]
    gt_vel           = [latents_choicemap(tr, t)[:vₜ => :val] for t in times]

    velret = plot_variable_over_time!(
        vel_layout,
        [sum(grid, dims=1) |> normalize |> to_vect for grid in posterior_probability_grids];
        times, groundtruth=gt_vel, varname="Velocity", varvals=Vels(),
        observations=nothing,
        show_medians=show_statistics, show_means=show_statistics, show_percentiles=show_statistics, markersize=15, show_colorbar=show_colorbars
    )

    posret = plot_variable_over_time!(
        pos_layout,
        [sum(grid, dims=2) |> normalize |> to_vect for grid in posterior_probability_grids];
        times,
        groundtruth=gt_pos,
        observations=pos_observations,
        varname="Position",
        varvals=Positions(),
        show_medians=show_statistics, show_means=show_statistics, show_percentiles=show_statistics, markersize=15, show_colorbar=show_colorbars
    )

    return (vel_layout, pos_layout, velret, posret)
end

function make_2d_posterior_figure(
    tr, posterior_probability_grids;
    inference_method_str=""
)    
    fig = Figure(resolution=(800, 1000))
    fig[1:2, :] = l = GridLayout()

    (_, _, (velax, _), (posax, (obs_plts, gt_plts, medplts, meanplts, per5plts, per95plts))) =
        draw_2d_posterior!(l, posterior_probability_grids, tr; show_statistics=true)
    
    rowsize!(l, 1, Relative(1/3))

    leg = Legend(fig[3, 1],
        [obs_plts, gt_plts, medplts, meanplts, per5plts, per95plts],
        ["Observation", "Ground truth", "Median under posterior", "Mean under posterior", "5th percentile under posterior", "95th percentile under posterior"]
    )
    leg.tellheight = true
    leg.tellwidth = false
    trim!(fig.layout)

    (fig[4, :] = Label(fig,
        "Obs Std = $(ObsStd()). Vel Step Std = $(VelStepStd()). Prob Vel Redrawn: $(SwitchProb())."
    )).tellwidth = false
    (fig[5, :] = Label(fig, inference_method_str)).tellwidth = false


    return fig
end

### Particle squares visualization ###

# pos_particles[t][pos] is a vector of pairs
# (weight, color) for each particle at that position at that time,
# giving the particle's weight and the color to use to draw it.
# The weights should be normalized (so the sum of weights should be 1).
# vel_particles has an analogous format.
# Particles are plotted in the square [time-0.5, time+0.5]x[pos-0.5, time+0.5]
# and similarly for velocities
function draw_particles!(layout, pos_particles, vel_particles, n_particles)
    velax = layout[1, 1] = Axis(layout[1, 1]; xlabel="Time", ylabel="Velocity")
    posax = layout[2, 1] = Axis(layout[2, 1]; xlabel="Time", ylabel="Position")
    rowsize!(layout, 1, Relative(1/3))
    draw_particles!(posax, velax, pos_particles, vel_particles, n_particles)
end
function draw_particles!(posax, velax, pos_particles, vel_particles, n_particles)
    draw_particle_squares_for_variable!(posax, Positions(), pos_particles, n_particles)
    draw_particle_squares_for_variable!(velax, Vels(), vel_particles, n_particles)
end
function draw_particle_squares_for_variable!(ax, varvals, time_to_particles, n_particles; starttime=0, T=(length(time_to_particles) - 1), size_scalar=1)
    particle_of_each_color = nothing
    for (t, val_to_particles) in zip(starttime:T, time_to_particles)
        for (val, particles) in zip(varvals, val_to_particles)
            p = _draw_particles!(ax, val, (t - 0.5, t + 0.5), particles, n_particles; size_scalar)
            if isnothing(particle_of_each_color) || isempty(particle_of_each_color)
                particle_of_each_color = p
            end
        end
    end
    # ax.xticks = (-0.5):(T+0.5)
    # ax.yticks = (first(varvals) - 0.5):(last(varvals) + 0.5)
    # ax.xticks = -1:(T+1)
    # ax.yticks = (first(varvals) - 1):(last(varvals) + 1)
    # ax.xgridvisible = false
    # ax.ygridvisible = false
    # ax.xminorgridvisible = true
    # ax.yminorgridvisible = true
    # ax.xminorticks = IntervalsBetween(2)
    # ax.yminorticks = IntervalsBetween(2)
    xlims!(ax, (starttime-0.5, T + 0.5))
    ax.xticks = starttime:T
    ylims!(ax, (first(varvals) - 0.5, last(varvals) + 0.5))

    return particle_of_each_color
end
function _draw_particles!(ax, pos, (leftmost_x, rightmost_x), particles, n_particles; size_scalar=1)
    max_padding = 0.1

    # TODO: improve the algorithm for how
    # the particles get drawn
    space_between_squares = 0.1 # max_padding / n_particles
    center = (leftmost_x + rightmost_x)/2
    # println("center = $center ; n_particles = $n_particles ; space_between_squares = $space_between_squares")
    current_x = center - ((n_particles - 1)/2 * space_between_squares)
    particle_of_each_color = []
    
    # sort the particles so we draw the most-likely particles first
    sorted_particles = sort(particles; by=(p -> p[1]), rev=true)
    for (weight, color) in sorted_particles
        # TODO: be more careful with the sizes?
        size = (1 - max_padding) * sqrt(weight) * size_scalar
        plt = draw_particle!(ax, pos, current_x, size,
            RGBA(convert(RGB, parse(Colorant, color)), weight)
        )
        push!(particle_of_each_color, plt)
        current_x += space_between_squares
    end

    return particle_of_each_color
end
draw_particle!(ax, ypos, startx, size, color) =
    # poly!(ax, [Rect(startx, ypos - sidelength/2, sidelength, sidelength)], color=color)
    scatter!(ax, [startx], [ypos], markersize=50*size, color=color)