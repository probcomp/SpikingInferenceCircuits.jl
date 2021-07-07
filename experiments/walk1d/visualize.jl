using GLMakie
set_theme!(colormap=:grays)

FRAMERATE() = 2 # frames per second

to_matrix(x) = reshape(onehot(x, Positions()), (:, 1))

visualize(time_to_x) = line_visualizations([to_matrix ∘ time_to_x])
function visualize(ax, t, time_to_matrix)
    ax.aspect = DataAspect()
    heatmap!(ax, @lift(time_to_matrix($t)), colorrange=(0, 1))
    hideydecorations!(ax)
    return ax
end
getobs(tr) = t -> obs_choicemap(tr, t)[:obs => :val]
getpos(tr) = t -> latents_choicemap(tr, t)[:xₜ => :val]
visualize_obs(tr) = visualize(getobs(tr))
visualize_pos(tr) = visualize(getpos(tr))

function line_visualizations(time_to_matrix_fns, titles=["" for _ in time_to_matrix_fns])
    t = Observable(0)
    fig = Figure()
    for (i, (f, title)) in enumerate(zip(time_to_matrix_fns, titles))
        ax = fig[i, 1] = Axis(fig; title)
        visualize(ax, t, f)
    end
    return (fig, t)
end

function animate(t, T)
    for _t = 0:T
        t[] = _t
        sleep(1/FRAMERATE())
    end
end

###
function obs_pos_enumerated_figure(tr)
    fig = Figure(); t = Observable(0)
    fig[1, 1] = visualize(Axis(fig; title="Pos"), t, to_matrix ∘ getpos(tr))
    fig[2, 1] = visualize(Axis(fig; title="Obs"), t, to_matrix ∘ getobs(tr))

    enumerated_weights = enumerate_latent_assmt_weights_from_groundtruth(
        tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),)
    ) |> nest_all_addrs_at_val |> collect
    
    inf_matrix(t) = reshape(
        exp.(enumerated_weights[t + 1]) |> normalize, (:, 1)
    )
    maxprob = maximum(maximum(inf_matrix(_t)) for _t=1:get_args(tr)[1])

    fig[3, 1] = infax = Axis(fig; title="Posterior Log-Probabilities")
    hm = heatmap!(infax, @lift(inf_matrix($t)), colorrange=(0, min(maxprob + 0.1, 1.0))
    )
    hideydecorations!(infax)

    # fig, t = line_visualizations([
    #         to_matrix ∘ getpos(tr),
    #         to_matrix ∘ getobs(tr),
    #         t -> reshape(exp.(enumerated_weights[t + 1]) |> normalize |> x->log.(x), (:, 1))
    #     ],
    #     ["Pos", "Obs", "Posterior"]
    # )
    Colorbar(fig[3, 2], hm)
    return (fig, t)
end