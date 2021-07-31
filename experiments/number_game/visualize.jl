using GLMakie

"""
- probvec = 100-long vector, giving, for each number, the inferred probability that the number is in the set
"""
make_hist!(ax, probs) = barplot!(
    ax, 1:length(probs), probs;
    color=:black, width=step(1:length(probs))
)
add_dots!(ax, pts) = scatter!(ax, pts, [0.02 for _ in pts]; color=:red, markersize=12)
function visualize(pts, probs; title="")
    f = Figure(resolution=(800,400))
    ax = Axis(f[1, 1], xlabel="Number", ylabel="Probability", title=title)
    make_hist!(ax, probs)
    add_dots!(ax, pts)
    xlims!(ax, (1, length(probs)))
    ylims!(ax, (0, 1))
    return f
end

unweighted_traces_to_probs(unweighted_traces) =
    weighted_traces_to_probs((tr, 1.) for tr in unweighted_traces)
weighted_traces_to_probs(weighted_trs) = 
    sum(
        weight * set_membership_vec(tree(trace))
        for (trace, weight) in weighted_trs
    )/sum(weight for (trace, weight) in weighted_trs)
set_membership_vec(tree) = [is_in_set(tree, x) ? 1. : 0. for x=1:100]

visualize_unweighted_traces(unweighted_traces; title="") =
    visualize(
        vals(first(unweighted_traces)),
        unweighted_traces_to_probs(unweighted_traces)
    )
visualize_weighted_traces(weighted_traces; title="") =
    visualize(
        vals(first(weighted_traces)[1]),
        weighted_traces_to_probs(weighted_traces); title
    )