#=
This is the code used to produce the 3D heatmap in the NeurIPS draft, which compared
expected latency to assembly size to the standard deviation of a probability estimate.
=#

using GLMakie

TR() = 1:100
AR() = 1:10

function get_stddev_grid(;
    max_rate=1.,
    largest_prob=0.5,
    assembly_size_range=AR(),
    TimeRange=TR(),
    p_i=0.3
)
    overall_rate(assembly_size) = max_rate * assembly_size / largest_prob
    n_trials(meantime, assembly_size) = meantime * overall_rate(assembly_size)
    variance(meantime, assembly_size) = p_i*(1-p_i) / n_trials(meantime, assembly_size)

    return [sqrt(variance(t, a)) for t in TimeRange, a in assembly_size_range]
end
function draw_stddev_surface(grid;
    assembly_size_range=AR(),
    TimeRange=TR()
)
    fig = Figure()
    ax = Axis3(
        fig[1, 1],
        aspect=(1, 1, 1),
        xlabel="E[Latency] (ms)",
        ylabel="Assembly Size",
        zlabel="Std.Dev. of Estimate",
        xlabelsize=40, ylabelsize=40, zlabelsize=40,
        xticklabelsize=25, yticklabelsize=25, zticklabelsize=25
        # scale=(identity, identity, log)
    )

    hm = surface!(
        ax, TimeRange, assembly_size_range, (x, y) -> grid[x,y]
    )
    # c = Colorbar(fig[1, 1][1, 2], hm, width=30, scale=log)

    # ax.xtickformat = xs -> ["$x" for x in xs]
    # ax.xlabel = "Expected Time to Estimate Probability (ms)"

    # ax.yticks = assembly_size_range
    # ax.ytickformat = ys -> ["$(Int(y))" for y in ys]
    # ax.ylabel = "Assembly size"

    # ax.title = "Log-Variance of Spike-Count Probability Estimate"

    # xticks!(ax, )
    # yticks!(ax, yticklabels=assembly_size_range)
    
    return fig
end

draw_stddev_surface(get_stddev_grid())

# get_probratio_grid(;
#     assembly_size_range=AR(),
#     rate_ratio_range=[10^x for x=1:0.2:6]
# ) = 
#     [
#         log(a * r)
#         for r in rate_ratio_range, a in assembly_size_range
#     ]
# function draw_probratio_surface(grid;
#     assembly_size_range=AR(),
#     rate_ratio_range=[10^x for x=1:0.2:6]
# )
#     fig = Figure()
#     ax = Axis3(
#         fig[1, 1],
#         aspect=(1, 1, 1),
#         xlabel="Max rate / min rate",
#         ylabel="Assembly Size",
#         zlabel="Dynamic range of probabilities",
#         xscale=log #,
#         # scale=(log, identity, log)
#     )

#     hm = surface!(
#         ax, (rate_ratio_range), assembly_size_range, *
#     )
#     # c = Colorbar(fig[1, 1][1, 2], hm, width=30, scale=log)

#     # ax.xtickformat = xs -> ["$x" for x in xs]
#     # ax.xlabel = "Expected Time to Estimate Probability (ms)"

#     # ax.yticks = assembly_size_range
#     # ax.ytickformat = ys -> ["$(Int(y))" for y in ys]
#     # ax.ylabel = "Assembly size"

#     # ax.title = "Log-Variance of Spike-Count Probability Estimate"

#     # xticks!(ax, )
#     # yticks!(ax, yticklabels=assembly_size_range)
    
#     return fig
# end

# # draw_probratio_surface(get_probratio_grid())