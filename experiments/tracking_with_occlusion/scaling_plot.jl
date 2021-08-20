using CairoMakie
# set_theme!(palette = (color = :seaborn_muted))
function make_plt(cpt_sizes, our_sizes)
    f = Figure()
    ax = Axis(
        f[1, 1];
        title="Neural Circuit Scaling in Model Size",
        xlabel="Number of variables (V)",
        ylabel="Number of neurons in ProbEstimate Circuit"
    )
    ours_empirical = scatter!(ax, map(x -> x^2, keys(our_sizes)), our_sizes, color=:blue)
    cpt_empirical = scatter!(ax, map(x -> x^2, keys(cpt_sizes)), cpt_sizes, color=:red, marker=:cross, markersize=20) #; marker='x')

    lin_factor = last(our_sizes) / (length(our_sizes)^2)
    ours_analytic = lines!(ax, 1:36, x -> lin_factor*x, color=:blue)
    cpt_analytic = lines!(ax, 1:10, x -> 3^x, color=:red)

    Legend(f[1, 2],
        [cpt_empirical, ours_empirical, cpt_analytic, ours_analytic],
        [
            "Probabilistic Population Coding",
            "Ours",
            "O(3^V)",
            "O(V)"
        ]
    )
    
    return f
end

# Neuron counts from running `scaling_data.jl`:
CPTSizes = [9, 168, 39386]
OurSizes = [14.0, 80.0, 231.0, 480.0, 825.0, 1296.0]
make_plt(CPTSizes, OurSizes)