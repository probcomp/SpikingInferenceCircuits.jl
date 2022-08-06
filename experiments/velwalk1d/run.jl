using DynamicModels
include("model.jl")
# include("pm_model.jl")
include("inference.jl")
include("visualize.jl")
ProbEstimates.DoRecipPECheck() = false
include("utils.jl")

# include("run_utils.jl")

# tr, _ = generate(model, (10,));
# make_smcexact_2d_posterior_figure(tr)
tr, _ = generate(model, (10,));
make_true_2d_posterior_figure(tr)
