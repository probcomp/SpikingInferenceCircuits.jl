# This file is a quick script to plot z estimates.
# It assumes variables `filenames` and `specs` are loaded into the local Julia environment.

includet("plot_z_estimates.jl")

# (gold, ests) = load_z_estimates_comparison(filename)
# plot_z_estimate_comparison_grid_v2(gold, reshape(ests, size(specs))[:, 2:2], xlabels[2:2], ylabels)

function plot_spec(filename, spec)
    (gold, ests) = load_z_estimates_comparison(filename)
    plot_z_estimate_comparison_grid_v2(gold, ests, [spec.constants_str], spec.labels; title="Z Estimates, Varying $(spec.varied_quantity)")
end
names_specs = collect(zip(filenames, specs))

f = plot_spec(names_specs[1]...)