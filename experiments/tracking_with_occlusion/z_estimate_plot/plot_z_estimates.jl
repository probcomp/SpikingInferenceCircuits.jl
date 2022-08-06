function scatter_z_ests!(ax, gold_standard_z_ests, ests, min_est=nothing, max_est=nothing)
    gold_standard_with_repeats = (
        [[gs for _ in estimates] for (gs, estimates) in zip(gold_standard_z_ests, ests)]
            |> Iterators.flatten |> collect
    )

    ### Scatter plot, which won't include values of -Inf ###
    ests_flat = ests |> Iterators.flatten |> collect
    if isnothing(min_est)
        min_est = [e for e in ests_flat if !isinf(e) && !isnan(e)] |> minimum
    end
    if isnothing(max_est)
        max_est = [e for e in ests_flat if !isinf(e) && !isnan(e)] |> maximum
    end
    non_inf_scatter = scatter!(ax, gold_standard_with_repeats, ests_flat, color=:black)

    ### Plot marks for all the values of -Inf ###
    original_limits = ax.elements[:yaxis].attributes.limits[]
    original_limits = ax.finallimits[]
    original_min_x, _ = original_limits.origin
    original_max_x, _ = original_limits.origin + original_limits.widths
    original_min_y, original_max_y = min_est - 20, max_est + 20

    # Now get the non-inf gold standard estimates for which -inf estimates occurred, and plot them.
    inf_est_gold_vals = [v for v in gold_standard_with_repeats[findall(x -> isinf(x) || isnan(x), ests_flat)] if !(isinf(v) || isnan(v))]
    inf_scatter = scatter!(ax,
        [Point2f0(x, original_min_y) for x in inf_est_gold_vals];
        marker=:x,
        color=:crimson
    )
    # TODO: after the `linkaxes!` call which occurs in `plot_z_estimate_comparison_grid_v2`,
    # these axes may change again.
    ylims!(ax, (original_min_y - 2, original_max_y))
    xlims!(ax, (original_min_x, original_max_x))

    return (non_inf_scatter, inf_scatter)
end

function plot_z_estimate_comparison(gold_standard_z_ests, z_estimates, labels)
    f = Figure()
    ax = Axis(f[1, 1], xlabel="Gold-Standard Log(P[yₜ | xₜ₋₁]) Estimate", ylabel="Log(P[yₜ | xₜ₋₁]) Estimate")
    plots = []
    for ests in z_estimates
        push!(plots, scatter_z_ests!(ax, gold_standard_z_ests, ests))
    end
    m = minimum(v for v in gold_standard_z_ests if !isinf(v)); M = maximum(gold_standard_z_ests)
    yx = lines!(ax, [Point2f0(m, m), Point2f0(M, M)])
    Legend(f[1, 2], [plots..., yx], [labels..., "y=x (0-variance z-estimate)"])
    return f
end

"""
Plot z estimates on seperate panes, each with their own title and axis labels.

`z_estimates` and `labels` should be Matrices of equal dimensionality;
the plot will have a grid of scatter plots
"""
function plot_z_estimate_comparison_grid(gold_standard_z_ests, z_estimates, labels)
    f = Figure()
    m = minimum(v for v in gold_standard_z_ests if !isinf(v)); M = maximum(gold_standard_z_ests)
    
    flattened_ests = z_estimates |> Iterators.flatten |> Iterators.flatten
    nonzero_flattened_ests = [v for v in flattened_ests if !isinf(v) && !isnan(v)]
    min_est = minimum(nonzero_flattened_ests); max_est = maximum(nonzero_flattened_ests)
    
    yx = nothing
    for (idx, label, ests) in zip(keys(labels), labels, z_estimates)
        (y, x) = Tuple(idx)
        ax = Axis(f[y, x], xlabel="Gold-Standard Log(P[yₜ | xₜ₋₁]) Estimate", ylabel="Log(P[yₜ | xₜ₋₁]) Estimate", title=label)
        scatter_z_ests!(ax, gold_standard_z_ests, ests, min_est, max_est)
        yx = lines!(ax, [Point2f0(m, m), Point2f0(M, M)])
    end
    ysize = size(labels)[1]
    Legend(f[ysize + 1, :], [yx], ["y=x (Perfect Log(P[yₜ | xₜ₋₁]) Estimate)"])
    return f
end

"""
Plot z estimates on a grid of axes, with shared column and row labels.
"""
function plot_z_estimate_comparison_grid_v2(
    gold_standard_z_ests, z_estimates,
    xlabels, ylabels;
    title=nothing
)
    f =
        if isnothing(title)
            Figure(;resolution=(800, 300 + 200*(length(ylabels))))
        else
            Figure(;title, resolution=(800, 300 + 200*(length(ylabels))))
        end
    grid = f[1, 1] = GridLayout()
    keys = CartesianIndices((length(ylabels), length(xlabels)))
    axs = [
        Axis(grid[y, x])
        for (y, x) in map(Tuple, keys)
    ]

    good_gold_ests = (v for v in gold_standard_z_ests if !isinf(v) && !isnan(v))
    m = minimum(good_gold_ests); M = maximum(good_gold_ests)

    flattened_ests = z_estimates |> Iterators.flatten |> Iterators.flatten
    nonzero_flattened_ests = [v for v in flattened_ests if !isinf(v) && !isnan(v)]
    min_est = minimum(nonzero_flattened_ests); max_est = maximum(nonzero_flattened_ests)

    yx = nothing; inf_val_plot = nothing;
    for (idx, ests) in zip(keys, z_estimates)
        (y, x) = Tuple(idx)
        ax = axs[y, x]
        (_, inf_val_plot) = scatter_z_ests!(ax, gold_standard_z_ests, ests, min_est, max_est)
        yx = lines!(ax, [Point2f0(m, m), Point2f0(M, M)]; color=:black)
    end

    for (x, xlabel) in enumerate(xlabels)
        axs[1, x].title = xlabel

        hidexdecorations!.(axs[1:(end-1), x], grid=false)
    end
    for (y, ylabel) in enumerate(ylabels)
        axs[y, 1].ylabel = ylabel

        hideydecorations!.(axs[y, 2:end], grid=false)
    end
    linkaxes!(axs...)
    
    Label(grid[end, :, Bottom()], "Gold-Standard Log(P[yₜ | xₜ₋₁]) Estimate", valign=:top, padding=(0, 0, 0, 30))
    Label(grid[:, 1, Left()], "Log(P[yₜ | xₜ₋₁]) Estimate", halign=:right, padding=(0, 100, 0, 0), rotation = pi/2)

    leg = Legend(f[2, 1],
        [ yx, inf_val_plot ],
        [ "y=x (Perfect Log(P[yₜ | xₜ₋₁]) Estimate)", "(Estimate of P[Yₜ | xₜ₋₁]) = 0" ]
    )
    leg.tellheight = true
    leg.tellwidth=false

    rowgap!(grid, 10)

    return f
end