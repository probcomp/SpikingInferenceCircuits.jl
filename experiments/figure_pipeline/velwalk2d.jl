module VelWalk2D
    using Revise: includet
    include("../velwalk2d_linked_vels/run.jl")
    gettr() = generate(model, (10,), choicemap(
            (:init => :latents => :xₜ => :val, 2),
            (:init => :latents => :yₜ => :val, 1),
            (:init => :latents => :vₜ => :val, (1, 2))
        )
    )[1]

    function figures(tr, callback)
        make_exact_filter_figure(tr)            |> callback(:ebf)
        make_smcexact_fig(tr, n_particles=10)   |> callback(:smcexact10)
        make_smcprior_fig(tr, n_particles=10)   |> callback(:prior10)
        make_smcprior_fig(tr, n_particles=1000) |> callback(:prior1000)
    end
end
