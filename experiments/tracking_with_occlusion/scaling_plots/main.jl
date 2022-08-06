using Distributions
using GLMakie

include("space_scaling/saved_scaling_data.jl")
include("variance_scaling/fracvars.jl")
include("representative_images.jl")
include("empirical_fracvars.jl")

set_theme!(font="Arial", palette = (color = [:red, :black],), linewidth=4, markersize=15, fontsize=20)

ColorFlipProb() = 0.001
default_hyperparams() = (
           latency        = 10,   # ms
           frequency      = 0.2, # spikes per ms
           neuron_budget = 500
       )

function space_scaling_plot!(f, pltpos, legpos; cpt_sizes=CPTSizes, our_sizes=OurSizes)
    ax = Axis(
        pltpos;
        xlabel="Number of Pixels",
        ylabel="Number of Neurons\nin Scoring Circuit",
        title="Scoring Circuit Size vs Image Size"
    )
    ours_empirical = scatter!(ax, map(x -> x^2, keys(our_sizes)), our_sizes, color=:black)
    cpt_empirical = scatter!(ax, map(x -> x^2, keys(cpt_sizes)), cpt_sizes, color=:red)#, marker=:cross, markersize=20)

    lin_factor = last(our_sizes) / (length(our_sizes)^2)
    ours_analytic = lines!(ax, 1:36, x -> lin_factor*x, color=:black)
    cpt_analytic = lines!(ax, 1:10, x -> 3^x, color=:red)

    leg = Legend(legpos,
        [cpt_empirical, ours_empirical, cpt_analytic, ours_analytic],
        [
            "Probabilistic Population Coding",
            "Ours",
            "O(3^V)",
            "O(V)"
        ]
    )
    leg.tellheight=true

    (ax, leg)
end

function singlepix_variance_scaling_plot!(plotpos; ps=0.0001:0.00001:0.05, hyperparams=default_hyperparams())
    k_aux = get_k(default_hyperparams(); use_aux=true)
    k_noaux = get_k(default_hyperparams(); use_aux=false)
    ax = Axis(plotpos,
        xlabel="Probability of Pixel Error",
        ylabel="Fractional Variance",
        title="Fractional Variance of Incorrect Pixel\n Probability Estimate (E[Latency] = $(hyperparams.latency)ms)",
        xscale=log10
    )
    noaux = lines!(ax, (ps), ([analytic_direct_fracvar(p, k_aux) for p in ps]))
    withaux = lines!(ax, (ps), ([analytic_prod_fracvar(sqrt(p), k_noaux) for p in ps]))
    return (ax, noaux, withaux)
end
function singlepix_latency_scaling_plot!(plotpos; probs=0.0001:0.00001:0.05, fracvar=0.5, hyperparams=default_hyperparams())
    ax = Axis(plotpos,
        xlabel="Probability of Pixel Error",
        ylabel="Expected Latency (ms)",
        title="Expected Latency to Score Incorrect Pixel\nat Fractional Variance $fracvar",
        xscale=log10
    )

    noaux_latencies = [k_to_Elatency(noaux_k_for_fracvar(p, fracvar), hyperparams.neuron_budget, hyperparams.frequency) for p in probs]
    withaux_latencies = [k_to_Elatency(withaux_k_for_fracvar(p, fracvar), hyperparams.neuron_budget, hyperparams.frequency) for p in probs]
    noaux = lines!(ax, (probs), noaux_latencies)
    withaux = lines!(ax, (probs), withaux_latencies)
    # ylims!(ax, (0, 400))

    return (ax, noaux, withaux)
end
function image_variance_scaling_plot!(plotpos; p_pix_flip=ColorFlipProb(), sidewidth=10, n_flips_range=1:10, hyperparams=default_hyperparams())
    ax = Axis(plotpos,
        xlabel="Number of Incorrect Pixels",
        ylabel="Fractional Variance",
        title="Fractional Variance of Image Likelihood\nvs Num Incorrect Pixels. (P[pixel error] = $p_pix_flip.)"
    )

    # k = get_k(default_hyperparams())
    # noaux_vals = exp.([analytic_logfracvar_for_num_pixelflips_noaux(p_pix_flip, n_flip, sidewidth^2, k) for n_flip in n_flips_range])
    # aux_vals = exp.([analytic_logfracvar_for_num_pixelflips_withaux(p_pix_flip, n_flip, sidewidth^2, k) for n_flip in n_flips_range])
    @assert ColorFlipProb() == p_pix_flip
    noaux_vals = image_likelihood_fracvars(hyperparams, sidewidth, n_flips_range, false, 10_000)
    aux_vals = image_likelihood_fracvars(hyperparams, sidewidth, n_flips_range, true, 10_000)
    noaux = lines!(ax, n_flips_range, noaux_vals)
    withaux = lines!(ax, n_flips_range, aux_vals)

    return (ax, noaux, withaux)
end
function auxvar_legend!(pos, (noaux, withaux))
    l = Legend(pos, [noaux, withaux], ["Simple Monte Carlo", "Spiking Neural Monte Carlo (Ours)"])
    l.tellheight = true
    l
end

function make_figure()
    f = Figure(;resolution=(2000, 600))
    space_scaling_plot!(f, f[1, 1], f[2, 1])

    singlepix_variance_scaling_plot!(f[1, 2])
    singlepix_latency_scaling_plot!(f[1, 3])
    (_, noaux, withaux) = image_variance_scaling_plot!(f[1, 4])
    auxvar_legend!(f[2, 1:4], (noaux, withaux))

    for (i, l) in zip(1:4, map(x -> "($x)", ["a", "b", "c", "d"]))
        Label(f.layout[1, i, BottomLeft()], l,
            textsize = 26,
            padding = (0, 0, 0, 25)
        )
    end

    # outer_layout = draw_representative_imgs_line!(f, (2, 3:4))
    # Label(outer_layout[1, 2, BottomLeft()], "(e)", textsize=26, padding=(0,0,0,25))

    # rowsize!(f.layout, 2, Relative(1/3))
    # f.layout[2, :].padding=(0,20,0,0)
    # rowgap!(f.layout, 60)

    f
end

f = make_figure()