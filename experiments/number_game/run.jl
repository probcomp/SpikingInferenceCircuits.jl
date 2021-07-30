using Gen, ProbEstimates, DynamicModels
ProbEstimates.use_perfect_weights!() # initially we will test this in vanilla Gen
include("model.jl")
include("inference.jl")

tree(tr) = tr[:init => :latents][1]
vals(tr) = [tr[:init => :obs][1], (tr[:steps => t => :obs][1] for t=1:get_args(tr)[1])...]

nums = [30, 31, 33, 24, 21, 36, 39]

tr, weight = generate(model, (length(nums) - 1,), choicemap(
    (:init => :obs => :number => :number => :val, nums[1]),
    (
        (:steps => t => :obs => :number => :number => :val, num)
        for (t, num) in enumerate(nums)
    )...
))
# (tree(tr), vals(tr))
resimulate_branch_cycle_ntimes(n) = n == 1 ? resimulate_branch_cycle : tr -> resimulate_branch_cycle_ntimes(n - 1)(resimulate_branch_cycle(tr))
(unweighted_trs, _) = dynamic_model_smc(model,
    get_dynamic_model_obs(tr),
    cm -> (cm[:number => :number => :val],),
    initial_proposal, step_proposal, 100;
    rejuvenate=resimulate_branch_cycle_ntimes(1)
);

