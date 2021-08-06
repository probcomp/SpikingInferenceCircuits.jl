using Gen, ProbEstimates, DynamicModels

ProbEstimates.MinProb() = 1/25
ProbEstimates.Latency() = 50.
ProbEstimates.AssemblySize() = 200
ProbEstimates.MaxRate() = 1.0
ProbEstimates.UseLowPrecisionMultiply() = false

use_ngf() = true
if use_ngf()
    ProbEstimates.use_noisy_weights!()
else
    ProbEstimates.use_perfect_weights!()
end

include("model.jl")
include("inference.jl")
include("enumeration.jl")
includet("visualize.jl")

tree(tr) = tr[:init => :latents][1]
vals(tr) = [tr[:init => :obs][1], (tr[:steps => t => :obs][1] for t=1:get_args(tr)[1])...]

# early_nums = [30, 31, 33, 24, 21, 36, 39]
# late_nums = [30, 33, 24, 21, 36, 31, 39]

# mode = :Early
# # nums = mode == :Late ? late_nums : early_nums
# nums = [33, 24, 21, 36, 31, 39, 30]

# obs_choicemap = choicemap(
#     (:init => :obs => :number => :number => :val, nums[1]),
#     (
#         (:steps => t => :obs => :number => :number => :val, num)
#         for (t, num) in enumerate(nums[2:end])
#     )...
# )

### Enumeration: ###
# membership_probs = get_number_membership_probs(obs_choicemap, length(nums) - 1)
# println("Inference via enumeration completed.")
# f = visualize(nums, membership_probs; title="P[# in set | observed nums ($mode sequence)] | maxdepth=$(MAXDEPTH()) | Enumeration Results")

### Resample-Move w/ Particle Gibbs: ###
# tr, weight = generate(model, (length(nums) - 1,), obs_choicemap)
# nparticles=100
# n_pgibbs_particles=2
# ncycles_per_step = 1
# function resimulate_branch_cycle_ntimes(n)
#     if n == 1
#         return tr -> resimulate_branch_cycle(tr, n_pgibbs_particles)
#     else
#         return tr -> resimulate_branch_cycle_ntimes(n - 1)(resimulate_branch_cycle(tr, n_pgibbs_particles))
#     end
# end
# (unweighted_trs, weighted_trs) = dynamic_model_smc(model,
#     get_dynamic_model_obs(tr),
#     cm -> (cm[:number => :number => :val],),
#     initial_proposal, step_proposal, nparticles;
#     rejuvenate=resimulate_branch_cycle_ntimes(ncycles_per_step)
# );
# end_weighted_traces = [(tr, exp(wt)) for (tr, wt) in last(weighted_trs)]
# println("Inference completed.")

ngf_str() = use_ngf() ? """
NeuralGen-Fast with MaxRate=$(ProbEstimates.MaxRate())Hz, As.Size=$(ProbEstimates.AssemblySize()), E[Latency]=$(ProbEstimates.Latency())ms.
$(ProbEstimates.UseLowPrecisionMultiply() ? "Resampling via low-precision single-line-compression multiplication." : "Resampling via auto-normalized multiplication (\"neural floating point\").")
""" : "Vanilla Gen."

get_title() = "" #="""
P[# in set | observed nums ($mode sequence)] | maxdepth=$(MAXDEPTH())
Resample-Move w/ Branch Resimulation PGibbs Rejuvenation.
$(ngf_str())
$nparticles particles; $n_pgibbs_particles PGibbs particles; ncycles per step = $ncycles_per_step.
"""=#

nums_to_filename(nums) = reduce(*, ["$(n)_" for n in nums]) * ".png"

numss = [
    [33, 24, 21, 36, 31, 39, 30],
    [24, 21, 36, 31, 39, 30, 33],
    [21, 36, 31, 39, 30, 33, 21],
    [36, 31, 39, 30, 33, 21, 21]
]

function do_inference_on_nums_and_save_fig(nums)
    obs_choicemap = choicemap(
        (:init => :obs => :number => :number => :val, nums[1]),
        (
            (:steps => t => :obs => :number => :number => :val, num)
            for (t, num) in enumerate(nums[2:end])
        )...
    )
    tr, weight = generate(model, (length(nums) - 1,), obs_choicemap)
    (unweighted_trs, weighted_trs) = dynamic_model_smc(model,
        get_dynamic_model_obs(tr),
        cm -> (cm[:number => :number => :val],),
        initial_proposal, step_proposal, nparticles;
        rejuvenate=resimulate_branch_cycle_ntimes(ncycles_per_step)
    );
    end_weighted_traces = [(tr, exp(wt)) for (tr, wt) in last(weighted_trs)]

    f = visualize_weighted_traces(end_weighted_traces; title=get_title())
    save(nums_to_filename(nums), f)
end
for nums in numss
    do_inference_on_nums_and_save_fig(nums)
end

# f = visualize_weighted_traces(end_weighted_traces; title=get_title())