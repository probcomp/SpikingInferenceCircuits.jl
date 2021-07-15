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
    if !isdir(dirname)
        mkdir(dirname)
    end
    return dirname
end

base = get_save_dir()

# TODO: could add in more calls to set hyperparameters between these
to_img_name(str::String) = length(split(str, ".")) == 2 ? str : str * ".png"
to_vid_name(str::String) = length(split(str, ".")) == 2 ? str : str * ".mp4"
to_img_name(sym::Symbol) = eval(sym) |> to_img_name
to_vid_name(sym::Symbol) = eval(sym) |> to_vid_name

include("walk1d.jl")
walk1d = dir(base, "walk1d")
Walk1D.ProbEstimates.use_perfect_weights!()
tr = Walk1D.gettr()
println("Walk1D trace generated.")
gen = dir(walk1d, "Gen")
Walk1D.figures(tr, name -> fig -> Walk1D.save(joinpath(gen, to_img_name(name)), fig))
println("Walk1D Gen figures created.")
Walk1D.ProbEstimates.use_noisy_weights!()
ngf = dir(walk1d, "NeuralGenFast")
Walk1D.figures(tr, name -> fig -> Walk1D.save(joinpath(ngf, to_img_name(name)), fig))
println("Walk1D NeuralGenFast figures created.")
println()

include("velwalk1d.jl")
VelWalk1D.ProbEstimates.use_perfect_weights!()
tr = VelWalk1D.get_tr_with_sharp_velchange()
println("VelWalk1D trace with a sharp velocity change generated.")
velwalk1d = dir(base, "velwalk1d")
gen = dir(velwalk1d, "Gen")
VelWalk1D.figures(tr, name -> fig -> VelWalk1D.save(joinpath(gen, to_img_name(name)), fig))
println("VelWalk1D Gen figures created.")
VelWalk1D.ProbEstimates.use_noisy_weights!()
ngf = dir(velwalk1d, "NeuralGenFast")
VelWalk1D.figures(tr, name -> fig -> VelWalk1D.save(joinpath(ngf, to_img_name(name)), fig))
println("VelWalk1D NeuralGenFast figures created.")

include("velwalk2d.jl")
VelWalk2D.ProbEstimates.DoRecipPECheck() = false
VelWalk2D.SwitchProb() = 0.
VelWalk2D.ProbEstimates.use_perfect_weights!()
velwalk2d = dir(base, "velwalk2d")
tr = VelWalk2D.gettr()
println("VelWalk2D trace generated.")
gen = dir(velwalk2d, "Gen")
VelWalk2D.figures(
    tr,
    name -> ((fig, t),) -> VelWalk2D.make_video(fig, t, 10, joinpath(gen, to_vid_name(name)))
)
println("VelWalk2D Gen figures created.")
VelWalk2D.ProbEstimates.use_noisy_weights!()
ngf = dir(velwalk2d, "NeuralGenFast")
VelWalk2D.figures(
    tr,
    name -> ((fig, t),) -> VelWalk2D.make_video(fig, t, 10, joinpath(ngf, to_vid_name(name)))
)
println("VelWalk2D NeuralGenFast figures created.")

VelWalk2D.SwitchProb() = 0.15
VelWalk2D.ProbEstimates.use_perfect_weights!()
velwalk2d = dir(base, "velwalk2d_with_velswitch")
tr = VelWalk2D.get_tr_with_sharp_velchange()
println("VelWalk2D trace generated with sharp velocity change.")
gen = dir(velwalk2d, "Gen")
VelWalk2D.figures(
    tr,
    name -> ((fig, t),) -> VelWalk2D.make_video(fig, t, 10, joinpath(gen, to_vid_name(name)))
)
println("VelWalk2D sharp velchange Gen figures created.")
VelWalk2D.ProbEstimates.use_noisy_weights!()
ngf = dir(velwalk2d, "NeuralGenFast")
VelWalk2D.figures(
    tr,
    name -> ((fig, t),) -> VelWalk2D.make_video(fig, t, 10, joinpath(ngf, to_vid_name(name)))
)
println("VelWalk2D sharp velchange NeuralGenFast figures created.")