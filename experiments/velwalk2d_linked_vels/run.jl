using DynamicModels

include("model.jl")
include("visualize.jl")
include("inference.jl")

ProbEstimates.DoRecipPECheck() = false
ProbEstimates.use_perfect_weights!()

model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 4)
@load_generated_functions()

domains() = (xₜ=Positions(), yₜ=Positions(), vxₜ=Vels(), vyₜ=Vels(), vₜ=Vels2D())

function get_enumeration_grids(tr)
    logweight_grids = enumeration_bayes_filter_from_groundtruth(
            tr, initial_latent_model, step_latent_model, obs_model, domains(),
            4 # first 2 addrs (xₜ, yₜ, vxₜ, vyₜ) are deterministic in the step model
        ) |> DynamicModels.nest_all_addrs_at_val;
    weight_grids = [exp.(logweight_grid) for logweight_grid in logweight_grids];
    return [
        sum(grid, dims=5)
        for grid in weight_grids
    ]
end
make_exact_filter_figure(tr) = make_exact_filter_figure(tr, get_enumeration_grids(tr))
make_exact_filter_figure(tr, grids) =
    vel_pos_plot(tr, grids; inference_str="exact Bayes filter.")

function x_vel_counts(trs, t)
    counts = Int[0 for _ in Positions(), _ in Positions(), _ in Vels(), _ in Vels()]
    for tr in trs
        counts[
            latents_choicemap(tr, t)[:xₜ => :val],
            latents_choicemap(tr, t)[:yₜ => :val],
            latents_choicemap(tr, t)[:vₜ => :val][1] - first(Vels()) + 1,
            latents_choicemap(tr, t)[:vₜ => :val][2] - first(Vels()) + 1
        ] += 1
    end
    return counts
end

x_counts(trs, t) = sum(x_vel_counts(trs, t), dims=(3, 4))
x_probs(trs, t) = x_counts(trs, t) |> normalize |> a->reshape(a, size(a)[1:2])

function make_smc_figure(smcfn, tr; n_particles, proposalstr)
    (unweighted_trs, _) = smcfn(tr, n_particles)
    probs = [x_vel_counts(unweighted_trs[t + 1], t) for t=0:(get_args(tr)[1])]
    vel_pos_plot(tr, probs; inference_str="$n_particles-particle SMC $proposalstr.")
end
make_smcprior_fig(tr; n_particles=1_000) =
    make_smc_figure(smc_from_prior, tr; n_particles, proposalstr="proposing from prior")
make_smcexact_fig(tr; n_particles=10) =
    make_smc_figure(smc_exact_proposal, tr; n_particles, proposalstr="proposing from exact posterior")
make_smcapprox_fig(tr; n_particles=10) =
    make_smc_figure(smc_approx_proposal, tr; n_particles, proposalstr="\nproposing from efficiently-encodable approximate posterior")

tr, _ = generate(model, (10,));
# println("Trace generated.")
fig, t = make_smcapprox_fig(tr, n_particles=10); fig

# grids = get_enumeration_grids(tr);
# println("Grids produced.")

# (fig, t) = make_exact_filter_figure(tr, grids); fig