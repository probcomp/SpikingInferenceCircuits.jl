using DynamicModels
include("model.jl")
include("groundtruth_rendering.jl")
include("proposal.jl")
include("visualize.jl")

ProbEstimates.use_perfect_weights!()

model = @DynamicModel(init_latent_model, step_latent_model, obs_model, 5)
init_prob = @compile_initial_proposal(_initial_proposal, 1)
step_prop = @compile_step_proposal(_step_proposal, 5, 1)

@load_generated_functions()

tr, _ = generate(model, (10,), choicemap(
	(:init => :latents => :xₜ => :val, 1),
	(:init => :latents => :vxₜ => :val, 2)
));

obs_choicemap_to_matrix(ch) =
	[
		ch[:img_inner => x => y => :got_photon => :val]
		for x=1:ImageSideLength(), y=1:ImageSideLength()
	]

unweighted_trs, _ = dynamic_model_smc(
    model, get_dynamic_model_obs(tr),
    cm -> (obs_choicemap_to_matrix(cm),),
    init_prob, step_prop, 10
);

(fig, t) = draw_gt_and_particles(tr, unweighted_trs); fig

# TODO: enumerate.  Performance will probably be an issue.
# Maybe using Marco's factor-graph library could help...but getting it set up
# will take a decent amount of work.
# domains() = (xₜ=SqPos(), yₜ=SqPos(), vxₜ=Vels(), vyₜ=Vels(), occ=OccPos())
# probs = enumeration_bayes_filter_from_groundtruth(
#     tr, init_latent_model, init_step_model, obs_model,
#     domains()
# ) |> DynamicModels.nest_all_addrs_at_val |> collectk