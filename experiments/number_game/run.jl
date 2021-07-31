using Gen, ProbEstimates, DynamicModels
ProbEstimates.use_perfect_weights!()
include("model.jl")
include("inference.jl")
includet("visualize.jl")

tree(tr) = tr[:init => :latents][1]
vals(tr) = [tr[:init => :obs][1], (tr[:steps => t => :obs][1] for t=1:get_args(tr)[1])...]

early_nums = [30, 31, 33, 24, 21, 36, 39]
late_nums = [30, 33, 24, 21, 36, 31, 39]

mode = :Late
nums = mode == :Late ? late_nums : early_nums

tr, weight = generate(model, (length(nums) - 1,), choicemap(
    (:init => :obs => :number => :number => :val, nums[1]),
    (
        (:steps => t => :obs => :number => :number => :val, num)
        for (t, num) in enumerate(nums)
    )...
))

nparticles=130
n_pgibbs_particles=2
resimulate_branch_cycle_ntimes(n) = n == 1 ? resimulate_branch_cycle : tr -> resimulate_branch_cycle_ntimes(n - 1)(resimulate_branch_cycle(tr, n_pgibbs_particles))
(unweighted_trs, weighted_trs) = dynamic_model_smc(model,
    get_dynamic_model_obs(tr),
    cm -> (cm[:number => :number => :val],),
    initial_proposal, step_proposal, nparticles;
    rejuvenate=resimulate_branch_cycle_ntimes(1)
);
end_weighted_traces = [(tr, exp(wt)) for (tr, wt) in last(weighted_trs)]
println("Inference completed.")

f = visualize_weighted_traces(end_weighted_traces; title="$nparticles particles. $n_pgibbs_particles PGibbs particles. Maxdepth = $(MAXDEPTH()). Late sequence.")
