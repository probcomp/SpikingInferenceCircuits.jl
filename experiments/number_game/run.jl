using Gen, ProbEstimates, DynamicModels
ProbEstimates.use_perfect_weights!()

include("model.jl")
include("inference.jl")
include("enumeration.jl")
includet("visualize.jl")

tree(tr) = tr[:init => :latents][1]
vals(tr) = [tr[:init => :obs][1], (tr[:steps => t => :obs][1] for t=1:get_args(tr)[1])...]

early_nums = [30, 31, 33, 24, 21, 36, 39]
late_nums = [30, 33, 24, 21, 36, 31, 39]

mode = :Late
nums = mode == :Late ? late_nums : early_nums

### Enumeration: ###
# obs_choicemap = choicemap(
#     (:init => :obs => :number => :number => :val, nums[1]),
#     (
#         (:steps => t => :obs => :number => :number => :val, num)
#         for (t, num) in enumerate(nums[2:end])
#     )...
# )
# membership_probs = get_number_membership_probs(obs_choicemap, length(nums) - 1)
# println("Inference via enumeration completed.")
# f = visualize(nums, membership_probs; title="P[# in set | observed nums ($mode sequence)] | maxdepth=$(MAXDEPTH()) | Enumeration Results")

### Resample-Move w/ Particle Gibbs: ###
tr, weight = generate(model, (length(nums) - 1,), obs_choicemap)
nparticles=400
n_pgibbs_particles=3
ncycles_per_step = 3
resimulate_branch_cycle_ntimes(n) = n == 1 ? resimulate_branch_cycle : tr -> resimulate_branch_cycle_ntimes(n - 1)(resimulate_branch_cycle(tr, n_pgibbs_particles))
(unweighted_trs, weighted_trs) = dynamic_model_smc(model,
    get_dynamic_model_obs(tr),
    cm -> (cm[:number => :number => :val],),
    initial_proposal, step_proposal, nparticles;
    rejuvenate=resimulate_branch_cycle_ntimes(ncycles_per_step)
);
end_weighted_traces = [(tr, exp(wt)) for (tr, wt) in last(weighted_trs)]
println("Inference completed.")

f = visualize_weighted_traces(end_weighted_traces;
    title="P[# in set | observed nums ($mode sequence)] | maxdepth=$(MAXDEPTH())\nResample-Move w/ Branch Resimulation PGibbs Rejuvenation\n$nparticles particles; $n_pgibbs_particles PGibbs particles; ncycles per step = $ncycles_per_step."
)