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

    maxprob(grid) = maximum(grid)
    hm = heatmap!(ax,
        Positions(), Positions(),
        @lift(probgrids[$t + 1]);
        colormap=cgrad([:white, :black], [0., 0.2, 1.0]),
        colorrange = @lift((0, min(maxprob(probgrids[$t + 1]), 1.0)))
    )
    fig[1, 2] = Colorbar(fig[1, 2], hm, label="Probability from Filtering")
    
    return (fig, t)
end

function plot_gt_obs!(ax, t, time_to_gt, time_to_obs)
    markersize = 30
    
    # Current point:
    scatter!(ax,
        @lift([time_to_gt($t)[1]]),
        @lift([time_to_gt($t)[2]]);
        color=:seagreen,
        markersize
    )

    # Obs:
    scatter!(ax,
        @lift([time_to_obs($t)[1]]),
        @lift([time_to_obs($t)[2]]);
        color=:gold,
        markersize = markersize * 2/3
    )

    # Previous 2 positions:
    scatter!(ax,
        @lift($t == 0 ? Int[] : [time_to_gt($t - 1)[1]]),
        @lift($t == 0 ? Int[] : [time_to_gt($t - 1)[2]]);
        color=RGBA(colorant"seagreen", 0.5),
        markersize
    )
    scatter!(ax,
        @lift($t < 2 ? Int[] : [time_to_gt($t - 2)[1]]),
        @lift($t < 2 ? Int[] : [time_to_gt($t - 2)[2]]);
        color=RGBA(colorant"seagreen", 0.25),
        markersize
    )


    lims = (first(Positions()) - 1, last(Positions()) + 1)
    xlims!(ax, lims); ylims!(ax, lims)

    return ax
end

time_to_pos(tr) = t -> (latents_choicemap(tr, t)[:xₜ => :val], latents_choicemap(tr, t)[:yₜ => :val])
time_to_obs(tr) = t -> (obs_choicemap(tr, t)[:obsx => :val], obs_choicemap(tr, t)[:obsy => :val])

plot_gt_obs(tr) = plot_gt_obs(time_to_pos(tr), time_to_obs(tr))

plot_gt_obs_probs(tr, probgrids) = plot_gt_obs_probs(time_to_pos(tr), time_to_obs(tr), probgrids)

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