module VelWalk1D
    using Revise: includet
    include("../velwalk1d/run.jl")

    gettr() = generate(model, (10,), choicemap(
        (:init => :latents => :xₜ => :val, last(Positions()) - 1),
        (:init => :latents => :vₜ => :val, -1)
    ))[1]

    get_tr_with_sharp_velchange() = VelWalk1D.generate(
        VelWalk1D.model, (10,), VelWalk1D.choicemap(
            (:init => :latents => :xₜ => :val, 16),
            (:init => :latents => :vₜ => :val, -2),
            (:steps => 1 => :latents => :vₜ => :val, -2),
            (:steps => 2 => :latents => :vₜ => :val, -2),
            (:steps => 3 => :latents => :vₜ => :val, -2),
            (:steps => 4 => :latents => :vₜ => :val, -2),
            (:steps => 5 => :latents => :vₜ => :val, 2),
            (:steps => 6 => :latents => :vₜ => :val, 2),
            (:steps => 7 => :latents => :vₜ => :val, 2),
            (:steps => 8 => :latents => :vₜ => :val, 2)
        )
    )[1]

    function figures(tr, callback)
        make_true_2d_posterior_figure(tr)                                 |> callback(:ebf)
        make_smcexact_2d_posterior_figure(tr, n_particles=10)             |> callback(:smcexact10)
        make_smcprior_2d_posterior_figure(tr, n_particles=10)             |> callback(:prior10)
        make_smcprior_2d_posterior_figure(tr, n_particles=1000)           |> callback(:prior1000)
        make_smc_prior_exactrejuv_2d_posterior_figure(tr, n_particles=10) |> callback(:priorgrejuv10)
    end
end
