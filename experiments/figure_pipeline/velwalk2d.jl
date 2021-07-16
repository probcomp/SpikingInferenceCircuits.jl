module VelWalk2D
    using Revise: includet
    include("../velwalk2d_linked_vels/run.jl")
    gettr() = generate(model, (10,), choicemap(
            (:init => :latents => :xₜ => :val, 2),
            (:init => :latents => :yₜ => :val, 1),
            (:init => :latents => :vₜ => :val, (1, 2))
        )
    )[1]
    get_tr_with_sharp_velchange() = generate(model, (10,), choicemap(
        (:init => :latents => :xₜ => :val, 2),
        (:init => :latents => :yₜ => :val, 1),
        (:init => :latents => :vₜ => :val, (1, 2)),
        (:steps => 1 => :latents => :vₜ => :val, (1, 2)),
        (:steps => 2 => :latents => :vₜ => :val, (1, 2)),
        (:steps => 3 => :latents => :vₜ => :val, (1, 2)),
        (:steps => 4 => :latents => :vₜ => :val, (1, 2)),
        (:steps => 5 => :latents => :vₜ => :val, (1, 2)),
        (:steps => 6 => :latents => :vₜ => :val, (-2, -2)),
        (:steps => 7 => :latents => :vₜ => :val, (-2, -2)),
        (:steps => 8 => :latents => :vₜ => :val, (-2, -2))
    ))[1]

    function figures(tr, callback)
        make_exact_filter_figure(tr)            |> callback(:ebf)
        make_smcexact_fig(tr, n_particles=10)   |> callback(:smcexact10)
        make_smcapprox_fig(tr, n_particles=10)  |> callback("smc_approx_10")
        make_smcprior_fig(tr, n_particles=10)   |> callback(:prior10)
        make_smcprior_fig(tr, n_particles=1000) |> callback(:prior1000)
    end
end
