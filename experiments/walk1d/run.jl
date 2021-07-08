includet("../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

includet("model.jl")
includet("visualize.jl")

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 1)
@load_generated_functions()

tr, _ = generate(model, (10,))

includet("enumerate_utils.jl")

enumerated_weights = enumerate_latent_assmt_weights_from_groundtruth(
    tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),)
) |> nest_all_addrs_at_val |> collect

normalize(pvec) = pvec/sum(pvec)
function obs_pos_enumerated_figure(tr)
fig, t = line_visualizations([
        to_matrix ∘ getpos(tr),
        to_matrix ∘ getobs(tr),
        t -> reshape(exp.(enumerated_weights[t + 1]) |> normalize |> x->log.(x), (:, 1))
    ],
    ["Pos", "Obs", "Posterior"]
)
make_video(fig, t, 9, "anim.mp4")