using GLMakie
set_theme!(colormap=:grays)

FRAMERATE() = 2 # frames per second

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