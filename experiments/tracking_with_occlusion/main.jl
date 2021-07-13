using DynamicModels
include("model.jl")
include("proposal.jl")

model = @DynamicModel(init_latent_model, step_latent_model, obs_model, 5)
init_prob = @compile_initial_proposal(_initial_proposal, 1)
step_prop = @compile_step_proposal(_step_proposal, 5, 1)

@load_generated_functions()

tr = simulate(model, (10,));

obs_choicemap_to_matrix(ch) =
	[
		ch[:img_inner => x => y => :got_photon => :val]
		for x=1:ImageSideLength(), y=1:ImageSideLength()
	]

unweighted_trs, _ = dynamic_model_smc(
    model, get_dynamic_model_obs(tr),
    obs_choicemap_to_matrix,
    init_prob, step_prop, 10
)

# TODO: enumerate.  Performance will probably be an issue.
# Maybe using Marco's factor-graph library could help...but getting it set up
# will take a decent amount of work.
# domains() = (xₜ=SqPos(), yₜ=SqPos(), vxₜ=Vels(), vyₜ=Vels(), occ=OccPos())
# probs = enumeration_bayes_filter_from_groundtruth(
#     tr, init_latent_model, init_step_model, obs_model,
#     domains()
# ) |> DynamicModels.nest_all_addrs_at_val |> collect