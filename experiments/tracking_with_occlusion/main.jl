using DynamicModels
# includet("model.jl")
# includet("groundtruth_rendering.jl")
includet("prior_proposal.jl")
# includet("visualize.jl")

ProbEstimates.use_perfect_weights!()

# model = @DynamicModel(init_latent_model, step_latent_model, obs_model, 5)
init_prob = @compile_initial_proposal(_initial_prior_proposal, 1)
step_prop = @compile_step_proposal(_step_prior_proposal, 5, 1)

@load_generated_functions()

# tr, _ = generate(model, (10,), choicemap(
# 	(:init => :latents => :xₜ => :val, 1),
# 	(:init => :latents => :vxₜ => :val, 2)
# ));

obs_choicemap_to_matrix(ch) =
	[
		ch[:img_inner => x => y => :pixel_color => :val]
		for x=1:ImageSideLength(), y=1:ImageSideLength()
	]

NParticles = 100
unweighted_trs, _ = dynamic_model_smc(
    model, get_dynamic_model_obs(tr),
    cm -> (obs_choicemap_to_matrix(cm),),
    init_prob, step_prop, NParticles
);

(fig, t) = draw_gt_and_particles(tr, unweighted_trs,
"$(length(first(unweighted_trs)))-particle SMC proposing from prior."
); fig

# TODO: enumerate.  Performance will probably be an issue.
# Maybe using Marco's factor-graph library could help...but getting it set up
# will take a decent amount of work.
# domains() = (xₜ=SqPos(), yₜ=SqPos(), vxₜ=Vels(), vyₜ=Vels(), occ=OccPos())
# probs = enumeration_bayes_filter_from_groundtruth(
#     tr, init_latent_model, init_step_model, obs_model,
#     domains()
# ) |> DynamicModels.nest_all_addrs_at_val |> collectk