# This file is a quick script to plot z estimates.
# It assumes variables `filenames` and `specs` are loaded into the local Julia environment.

includet("plot_z_estimates.jl")

# (gold, ests) = load_z_estimates_comparison(filename)
# plot_z_estimate_comparison_grid_v2(gold, reshape(ests, size(specs))[:, 2:2], xlabels[2:2], ylabels)
# global gold = nothing
# global ests = nothing
# function plot_spec(filename, spec)
#     (g, e) = load_z_estimates_comparison(filename)
#     global gold = g
#     global ests = e
#     gold_vals = map(first, g)
#     est_vals = [[map(first, y) for y in x] for x in e]
# end
names_specs = collect(zip(filenames, specs))

function load(names_specs, i)
    (filename, spec) = names_specs[i]
    (g, e) = load_z_estimates_comparison(filename)
    return (g, e, spec)
end
plot(gold, ests, spec) =
    plot_z_estimate_comparison_grid_v2(
        map(first, gold),
        [[map(first, y) for y in x] for x in ests],
        [spec.constants_str],
        spec.labels; title="Z Estimates, Varying $(spec.varied_quantity)"
    )
plot(ges::Tuple) = plot(ges...)

(gold, ests, spec) = ges = load(names_specs, 1);
plot(ges)

# to visualize the particles inferred when making z estimates
# for a given datapoint, run a line like
# visualize_closest_datapoint(1, gold, ests, (-18, -Inf))
# here, `1` is since we are looking at a datapoint in the first (top) plot;
# `(-18, -Inf)` means to inspect a datapoint with position around `(-18, -Inf)`