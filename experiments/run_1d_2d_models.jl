using Base.Filesystem
import Dates

#=
Status of proposals having acceptable branching factor:
[x] 1D walk
[ ] 1D vel
[ ] 2D vel
=#

const ebf           = "exact_bayes_filter"
const prior10       = "smc_prior_10"
const prior1000     = "smc_prior_1000"
const smcexact10    = "smc_exact_10"
const priorgrejuv10 = "smc_prior_gibbsrejuv_10"

# TODO: could add constraints on traces
module Walk1D
    using Revise: includet
    include("walk1d/run.jl")
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
module VelWalk1D
    using Revise: includet
    include("velwalk1d/run.jl")
    gettr() = generate(model, (10,), choicemap(
        (:init => :latents => :xₜ => :val, last(Positions()) - 1),
        (:init => :latents => :vₜ => :val, -1)
    ))[1]

    function figures(tr, callback)
        make_true_2d_posterior_figure(tr)                                 |> callback(:ebf)
        make_smcexact_2d_posterior_figure(tr, n_particles=10)             |> callback(:prior10)
        make_smcprior_2d_posterior_figure(tr, n_particles=10)             |> callback(:prior1000)
        make_smcprior_2d_posterior_figure(tr, n_particles=1000)           |> callback(:smcexact10)
        make_smc_prior_exactrejuv_2d_posterior_figure(tr, n_particles=10) |> callback(:priorgrejuv10)
    end
end
module VelWalk2D
    using Revise: includet
    include("velwalk2d_linked_vels/run.jl")
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

function get_save_dir()
    basedir=joinpath(
        #  src/SpikingInferenceCircuits.jl              src/         
        Base.find_package("SpikingInferenceCircuits") |> dirname |> dirname,
        "experiments/figure_saves"
    )
    if !isdir(basedir)
        mkdir(basedir)
    end
    time = Dates.format(Dates.now(), "yyyy-mm-dd__HH-MM")
    dir = joinpath(basedir, time)
    if isfile(dir) || isdir(dir)
        time = Dates.format(Dates.now(), "yyyy-mm-dd__HH-MM-SS")
        dir = joinpath(basedir, time)
    end
    mkdir(dir)

    return dir
end
function dir(base, extension)
    dirname = joinpath(base, extension)
    mkdir(dirname)
    return(dirname)
end

base = get_save_dir()

# TODO: could add in more calls to set hyperparameters between these
to_img_name(str::String) = length(split(str, ".")) == 2 ? str : str * ".png"
to_vid_name(str::String) = length(split(str, ".")) == 2 ? str : str * ".mp4"
to_img_name(sym::Symbol) = eval(sym) |> to_img_name
to_vid_name(sym::Symbol) = eval(sym) |> to_vid_name

# walk1ddir = dir(base, "walk1d")
# Walk1D.figures(Walk1D.gettr(), name -> fig -> Walk1D.save(joinpath(walk1ddir, to_img_name(name)), fig))

# velwalk1ddir = dir(base, "velwalk1d")
# VelWalk1D.figures(VelWalk1D.gettr(), name -> fig -> VelWalk1D.save(joinpath(velwalk1ddir, to_img_name(name)), fig))

velwalk2ddir = dir(base, "velwalk2d")
VelWalk2D.figures(
    VelWalk2D.gettr(),
    name -> ((fig, t),) -> VelWalk2D.make_video(fig, t, 10, joinpath(velwalk2ddir, to_vid_name(name)))
)