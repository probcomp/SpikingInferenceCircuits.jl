includet("../../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

includet("model.jl")
includet("../visualize.jl")

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 2)
@load_generated_functions()

includet("../enumerate_utils.jl")

function get_x_probs(tr)
    logweight_grids = enumerate_latent_assmt_weights_from_groundtruth(
            tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),vₜ=Vels())
        ) |> nest_all_addrs_at_val
    weight_grids = [exp.(logweight_grid) for logweight_grid in logweight_grids]
    weight_vecs = [sum(grid, dims=2) for grid in weight_grids]
    return [normalize(weights) for weights in weight_vecs]
end

make_true_2d_posterior_figure(tr) = make_2d_posterior_figure(tr, get_x_probs(tr);
    inference_method_str="Posterior from exact Bayes filter."
)

tr, _ = generate(model, (10,)); make_true_2d_posterior_figure(tr)