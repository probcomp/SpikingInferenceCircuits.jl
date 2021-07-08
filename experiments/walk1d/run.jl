includet("../../src/DynamicModels/DynamicModels.jl")
using .DynamicModels

includet("model.jl")
include("inference.jl")
includet("visualize.jl")

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 1)
@load_generated_functions()

tr, _ = generate(model, (10,))

includet("enumerate_utils.jl")

function obs_pos_enumerated_figure(tr)
    enumerated_weights = enumerate_latent_assmt_weights_from_groundtruth(
        tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),)
    ) |> nest_all_addrs_at_val |> collect

    (fig, t) = obs_pos_inferences_figure(tr, [(
        "Exact posterior probabilities",
        t -> exp.(enumerated_weights[t + 1]) |> normalize
    )])
    
    return (fig, t)
end

function x_counts(trs, t)
    counts = Int[0 for _ in Positions()]
    for tr in trs
        counts[latents_choicemap(tr, t)[:xₜ => :val]] += 1
    end
    return counts
end
function obs_pos_priorsmc_figure(tr; n_particles=1_000)
    (unweighted_trs, _) = smc_from_prior(tr, n_particles)
    
    (fig, t) = obs_pos_inferences_figure(tr, [(
        "Inferred Probabilities via SMC from Prior with $n_particles Particles & full resampling",
        t -> x_counts(unweighted_trs[t + 1], t) |> normalize
    )])
    
    return (fig, t)
end

# make_video(fig, t, 9, "anim.mp4")

# fig, t = obs_pos_priorsmc_figure(tr); fig

obs_pos_enumerated_figure(tr)