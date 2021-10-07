include("model.jl")
include("groundtruth_rendering.jl")
include("visualize.jl")
include("obs_aux_proposal.jl")
include("prior_proposal.jl")
include("nearly_locally_optimal_proposal.jl")
include("run_utils.jl")
include("flux_proposal.jl")

use_ngf() = false
if use_ngf()
    ProbEstimates.use_noisy_weights!()
else
    ProbEstimates.use_perfect_weights!()
end

include("z_estimates.jl")

step_proposal_flux = @compile_step_proposal(flux_proposal_MAP, obs_aux_proposal, 5, 1)
@load_generated_functions()



# ### Run inference:
# do_inference(tr; n_particles=10) = dynamic_model_smc(
#     model, get_returned_obs(tr),
#     cm -> (obs_choicemap_to_vec_of_vec(cm),),
#     initial_near_locopt_proposal, step_near_locopt_proposal, n_particles
# );


do_inference(tr; n_particles=10) = dynamic_model_smc(
    model, get_returned_obs(tr),
    cm -> (obs_choicemap_to_vec_of_vec(cm),),
    initial_near_locopt_proposal, step_proposal_flux, n_particles
);

# # ## Script to run inference + make visualizations
VelOneOffProb() = 0.1
gt_tr = generate_occluded_bounce_tr();
(unweighted_trs, weighted_trs) = do_inference(gt_tr; n_particles=30);
(fig, t) = make_gt_particle_viz_img_only(gt_tr, unweighted_trs); fig
display(fig)
animate(t, get_args(gt_tr)[1])
















# # Inference results animation:

# t[] = 2
# save("inferenceframe1.png", fig)
# t[] = 4
# save("inferenceframe2.png", fig)
# t[] = 6
# save("inferenceframe3.png", fig)

# (obsfig, tobs) = draw_obs(gt_tr); obsfig
# tobs[] = 6
# save("obs.png", obsfig)

# # Spiketrain figure:
# f = make_spiketrain_fig(
#     last(unweighted_trs)[1], 1:3; nest_all_at=(:steps => 1 => :latents),
#     resolution=(600, 450), figure_title="Dynamically Weighted Spike Code from Inference"
# )

# Draw observations:
# tr, _= generate(model, (2,), choicemap(
#     (:init => :latents => :xₜ => :val, 6),
#     (:init => :latents => :yₜ => :val, 6),
#     (:init => :latents => :occₜ => :val, 7)
# ))
# using CairoMakie
# CairoMakie.activate!()
# (fig, t) = draw_obs(tr); fig

