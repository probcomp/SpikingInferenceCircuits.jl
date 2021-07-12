using GLMakie, Colors

FRAMERATE() = 2

function plot_gt_obs(time_to_gt, time_to_obs)
    fig = Figure()
    ax = Axis(fig[1, 1])
    t = Observable(0)

    plot_gt_obs!(ax, t, time_to_gt, time_to_obs)
    
    return (fig, t)
end

function plot_gt_obs_probs(time_to_gt, time_to_obs, probgrids)
    fig = Figure()
    ax = Axis(fig[1, 1])
    t = Observable(0)

    plot_gt_obs!(ax, t, time_to_gt, time_to_obs)
    plot_probs!(ax, t, fig, (1, 2), probgrids, Positions())
    
    return (fig, t)
end

### Single plot with points & probs ###

function plot_probs!(ax, t, fig, colorbar_location, probgrids, domain)
    hm = heatmap!(ax,
        domain, domain,
        @lift(probgrids[$t + 1]);
        colormap=cgrad([:white, :black], [0, 0.2, 1.0]),
        colorrange = @lift((0, min(maximum(probgrids[$t + 1]), 1.0)))
    )
    fig[colorbar_location...] = Colorbar(
        fig[colorbar_location...], hm, label="Probability from Filtering"
    )
    return hm
end

function plot_points!(ax, t, time_to_pt, domain;
    n_backtrack=0, color, markersize=30, marker=:circle
)
    plts = []
    # Current point:
    scatter!(ax,
        @lift([time_to_pt($t)[1]]),
        @lift([time_to_pt($t)[2]]);
        color, markersize, marker
    ) |> plt -> push!(plts, plt)
    
    # previous points
    for i=1:n_backtrack
        scatter!(ax,
            @lift($t < i ? Int[] : [time_to_pt($t - i)[1]]),
            @lift($t < i ? Int[] : [time_to_pt($t - i)[2]]);
            color=RGBA(color, 0.6^i),
            markersize = markersize * 0.75^i,
            marker
        ) |> plt -> push!(plts, plt)
    end

    lims = (first(domain) - 0.5, last(domain) + 0.5)
    xlims!(ax, lims); ylims!(ax, lims)

    return plts
end

function plot_gt_obs!(ax, t, time_to_gt, time_to_obs)
    obs = plot_points!(ax, t, time_to_obs, Positions(); n_backtrack=2, color=colorant"gold", marker=:rect)
    gt = plot_points!(ax, t, time_to_gt, Positions(); n_backtrack=2, color=colorant"seagreen")
    return (gt, obs)
end

### Wrappers ###

time_to_pos(tr) = t -> (latents_choicemap(tr, t)[:xₜ => :val], latents_choicemap(tr, t)[:yₜ => :val])
time_to_obs(tr) = t -> (obs_choicemap(tr, t)[:obsx => :val], obs_choicemap(tr, t)[:obsy => :val])
time_to_vel(tr) = t -> latents_choicemap(tr, t)[:vₜ => :val]

plot_gt_obs(tr) = plot_gt_obs(time_to_pos(tr), time_to_obs(tr))

plot_gt_obs_probs(tr, probgrids) = plot_gt_obs_probs(time_to_pos(tr), time_to_obs(tr), probgrids)

### Plot with Vel & Pos ###
sum_grids(grids, sumover, reshapeto) =
    [sum(grid, dims=sumover) |> normalize |> g->reshape(g, size(g)[reshapeto]) for grid in grids]
function vel_pos_plot(tr, grids; inference_str="")
    fig = Figure(resolution=(800, 1000))
    t = Observable(0)

    velax = Axis(fig[1, 1], title="Velocity")
    plot_probs!(velax, t, fig, (1, 2),
        sum_grids(grids, (1, 2), 3:4),
        Vels()
    )
    plot_points!(velax, t, time_to_vel(tr), Vels();
        n_backtrack=2, color=colorant"seagreen"
    )

    posax = Axis(fig[2, 1], title="Position")
    plot_probs!(posax, t, fig, (2, 2),
        sum_grids(grids, (3, 4), 1:2),
        Positions()
    )
    (gt, obs) = plot_gt_obs!(posax, t, time_to_pos(tr), time_to_obs(tr))

    rowsize!(fig.layout, 1, Relative(1/3))
    velax.aspect = DataAspect()

    l = Legend(fig[3, 1], [gt, obs], ["Ground truth", "Observation"])
    l.tellheight = true
    l.tellwidth = false

    # hyperparameters
    fig[4, :] = hyperparam_label = Label(fig,
        "Positions = $(Positions()). Vels = $(Vels()).\nVelStepStd = $(VelStepStd()). ObsStd = $(ObsStd()). VelSwitchProb = $(SwitchProb())."
    )
    hyperparam_label.tellwidth = false

    fig[5, :] = inference_label = Label(fig, inference_str)
    inference_label.tellwidth = false

    return (fig, t)
end

### Animation ###

function animate(t, T)
    for _t = 0:T
        t[] = _t
        sleep(1/FRAMERATE())
    end
end

make_video(fig, t, T, filename) =
    record(fig, filename, 0:T; framerate=FRAMERATE()) do _t
        t[] = _t
    end