module Walk1D
    using Revise: includet
    include("../walk1d/run.jl")
    middleval = floor((first(Positions()) + last(Positions()))/2) |> Int
    gettr() = generate(model, (10,), choicemap((:init => :latents => :xₜ => :val, middleval)))[1]

    function figures(tr, callback)
        make_true_2d_posterior_figure(tr)                                 |> callback(:ebf)
        make_smcprior_2d_posterior_figure(tr, n_particles=10)             |> callback(:prior10)
        make_smcprior_2d_posterior_figure(tr, n_particles=1000)           |> callback(:prior1000)
        make_smcexact_2d_posterior_figure(tr, n_particles=10)             |> callback(:smcexact10)
        make_smc_prior_exactrejuv_2d_posterior_figure(tr, n_particles=10) |> callback(:priorgrejuv10)
    end    
end
