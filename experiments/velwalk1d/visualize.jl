using GLMakie

### 2D Visualizations ###
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

function time_variable_figure(
    fig, i, layout, posterior_probvec_at_times;
    times, groundtruth, observations=nothing, varname, varvals, markersize=15
)
    medians      = map(median(varvals), posterior_probvec_at_times) 
    means        = map(mean(varvals)  , posterior_probvec_at_times) 

    layout[1, 1] = ax = Axis(fig[1, 1], xlabel="Time", ylabel=varname)
    inf_matrix = hcat(posterior_probvec_at_times...) |> transpose
    maxprob = maximum(inf_matrix)
    hm = heatmap!(ax,
        times, varvals,
        inf_matrix,
        colormap=cgrad([:white, :black], [0., 0.4, 1.0]),
        colorrange = (0, min(maxprob + 0.1, 1.0))
    )
    layout[1, 2] = Colorbar(fig[i, 2], hm, label="Posterior probability")

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
    medplt(f) = f(ax,
        [t for t in times],
        [med for med in medians];
        color=:navy,
        markersize,
        (f == scatter! ? (marker=:star4,) : ())...
    )
    medplts = [medplt(lines!)] #, medplt(scatter!)]

    # Means:
    meanplt(f) = f(ax,
        [t for t in times],
        [m for m in means];
        color=:black,
        (f == scatter! ? (marker=:hline,) : ())...
    )
    meanplts = [meanplt(lines!)]

    # percentiles:
    percplt(f, p) = f(ax,
        [t for t in times],
        [percentile(varvals)(pvec, p) for pvec in posterior_probvec_at_times],
        color=:crimson
    )
    per5plts = [percplt(lines!, 0.05)]
    per95plts = [percplt(lines!, 0.95)]

    return (ax, (obs_plts, gt_plts, medplts, meanplts, per5plts, per95plts))
end

function make_2d_posterior_figure(
    tr, posterior_probability_grids;
    inference_method_str=""
)
    times = 0:(get_args(tr)[1])
    pos_observations = [obs_choicemap(tr, t)[:obs => :val] for t in times]
    gt_pos           = [latents_choicemap(tr, t)[:xₜ => :val] for t in times]
    gt_vel           = [latents_choicemap(tr, t)[:vₜ => :val] for t in times]
    
    fig = Figure(resolution=(800, 1000))
    fig[1:2, :] = l = GridLayout()
    vel_layout = GridLayout(); pos_layout = GridLayout()
    l[1, :] = vel_layout; l[2, :] = pos_layout

    velax, _ = time_variable_figure(
        fig, 1, vel_layout,
        [sum(grid, dims=1) |> normalize |> to_vect for grid in posterior_probability_grids];
        times,
        groundtruth=gt_vel,
        observations=nothing,
        varname="Velocity",
        varvals=Vels()
    )
    posax, (obs_plts, gt_plts, medplts, meanplts, per5plts, per95plts) = time_variable_figure(
        fig, 2, pos_layout,
        [sum(grid, dims=2) |> normalize |> to_vect for grid in posterior_probability_grids];
        times,
        groundtruth=gt_pos,
        observations=pos_observations,
        varname="Position",
        varvals=Positions()
    )

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