using GLMakie
set_theme!(colormap=:grays)

FRAMERATE() = 2 # frames per second

### 1D visualizations ###

to_matrix(x) = reshape(onehot(x, Positions()), (:, 1))

function visualize_state!(ax, t, time_to_x)
    ax.aspect = DataAspect()
    heatmap!(ax, @lift(to_matrix(time_to_x($t))), colorrange=(0, 1))
    hideydecorations!(ax)
    return ax
end
function visualize_inference!(fig, figpos, t, maxt, time_to_pvec; title)
    inf_matrix(_t) = reshape(time_to_pvec(_t), (:, 1))
    maxprob = maximum(maximum(inf_matrix(_t)) for _t=1:maxt)

    ax = fig[figpos, 1] = Axis(fig; title)
    hm = heatmap!(ax, @lift(inf_matrix($t)), colorrange=(0, min(maxprob + 0.1, 1.0)))
    hideydecorations!(ax)

    Colorbar(fig[figpos, 2], hm)

    return ax
end

getobs(tr) = t -> obs_choicemap(tr, t)[:obs => :val]
getpos(tr) = t -> latents_choicemap(tr, t)[:xₜ => :val]

function obs_pos_inferences_figure(tr, title_inferencefn_pairs)
    fig = Figure(); t = Observable(0)
    fig[1, 1] = visualize_state!(Axis(fig; title="Pos"), t, getpos(tr))
    fig[2, 1] = visualize_state!(Axis(fig; title="Obs"), t, getobs(tr))

    for (i, (title, time_to_pvec)) in enumerate(title_inferencefn_pairs)
        visualize_inference!(fig, i+2, t, get_args(tr)[1], time_to_pvec; title)
    end

    return fig, t
end


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

### 2D Visualizations ###
median(pvec) = percentile(pvec, 0.5)
function percentile(pvec, percentile)
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
mean(pvec) = sum(p * i for (i, p) in enumerate(pvec))
function make_2d_posterior_figure(
    tr, posterior_probvec_at_times;
    inference_method_str=""
)
    times = 0:(get_args(tr)[1])
    observations = map(getobs(tr), times)
    true_x       = map(getpos(tr), times)
    medians      = map(median, posterior_probvec_at_times) 
    means        = map(mean  , posterior_probvec_at_times) 
    
    fig = Figure()
    main_layout = GridLayout()
    fig[1, :] = main_layout
    main_layout[1, 1] = ax = Axis(fig[1, 1], xlabel="Time", ylabel="Position", xticklabels=-1:(get_args(tr)[1] - 1))
    
    inf_matrix = hcat(posterior_probvec_at_times...) |> transpose
    maxprob = maximum(inf_matrix)
    hm = heatmap!(ax,
        inf_matrix,
        colormap=cgrad([:white, :black], [0., 0.4, 1.0]),
        colorrange = (0, min(maxprob + 0.1, 1.0))
    )
    main_layout[1, 2] = Colorbar(fig[1, 2], hm, label="Posterior probability")

    markersize = 15.0
    # Observations:
    obsplt(f) = f(ax,
        [t+1 for t in times],
        [o for o in observations];
        color=:gold,
        markersize
    )
    obs_plts = [obsplt(lines!), obsplt(scatter!)]

    # Ground truth pos:
    gtplt(f) = f(ax,
        [t+1 for t in times],
        [tx for tx in true_x];
        color=:seagreen,
        markersize
    )
    gt_plts = [gtplt(lines!), gtplt(scatter!)]
    
    # Medians:
    medplt(f) = f(ax,
        [t+1 for t in times],
        [med for med in medians];
        color=:navy,
        markersize,
        (f == scatter! ? (marker=:star4,) : ())...
    )
    medplts = [medplt(lines!)] #, medplt(scatter!)]

    # Means:
    meanplt(f) = f(ax,
        [t+1 for t in times],
        [m for m in means];
        color=:black,
        (f == scatter! ? (marker=:hline,) : ())...
    )
    meanplts = [meanplt(lines!)] #, meanplt(scatter!)]

    # percentiles:
    percplt(f, p) = f(ax,
        [t + 1 for t in times],
        [percentile(pvec, p) for pvec in posterior_probvec_at_times],
        color=:crimson
    )
    per5plts = [percplt(lines!, 0.05)]
    per95plts = [percplt(lines!, 0.95)]

    leg = Legend(fig[2, 1],
        [obs_plts, gt_plts, medplts, meanplts, per5plts, per95plts],
        ["Observation", "True Position", "Median position under posterior", "Mean position under posterior", "5th percentile position under posterior", "95th percentile position under posterior"]
    )
    leg.tellheight = true
    leg.tellwidth = false
    trim!(fig.layout)

    (fig[3, :] = Label(fig,
        "Obs Std = $(ObsStd()), Step Std = $(StepStd())."
    )).tellwidth = false
    (fig[4, :] = Label(fig, inference_method_str)).tellwidth = false


    return fig
end