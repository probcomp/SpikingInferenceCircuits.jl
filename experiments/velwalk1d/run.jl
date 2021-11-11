using DynamicModels
include("model.jl")
# include("pm_model.jl")
include("inference.jl")
include("visualize.jl")
ProbEstimates.DoRecipPECheck() = false
include("utils.jl")

tr, _ = generate(model, (10,));
make_true_2d_posterior_figure(tr)