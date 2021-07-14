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

# include("walk1d.jl")
# walk1ddir = dir(base, "walk1d")
# Walk1D.figures(Walk1D.gettr(), name -> fig -> Walk1D.save(joinpath(walk1ddir, to_img_name(name)), fig))

include("velwalk1d.jl")
VelWalk1D.ProbEstimates.use_perfect_weights!()
println("Turned on perfect weights.")
tr = VelWalk1D.gettr()
println("Generated a trace.")
velwalk1ddir = dir(base, "velwalk1d_Gen")
VelWalk1D.figures(tr, name -> fig -> VelWalk1D.save(joinpath(velwalk1ddir, to_img_name(name)), fig))
println("Produced figures.")

VelWalk1D.ProbEstimates.use_noisy_weights!()
println("Turned on noisy weights.")
velwalk1d_noisydir = dir(base, "velwalk1d_NeuralGenFast")
VelWalk1D.figures(tr, name -> fig -> VelWalk1D.save(joinpath(velwalk1d_noisydir, to_img_name(name)), fig))
println("Produced figures.")

# include("velwalk2d.jl")
# velwalk2ddir = dir(base, "velwalk2d")
# VelWalk2D.figures(
#     VelWalk2D.gettr(),
#     name -> ((fig, t),) -> VelWalk2D.make_video(fig, t, 10, joinpath(velwalk2ddir, to_vid_name(name)))
# )