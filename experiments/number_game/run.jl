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

### Inference util functions: ###

function resimulate_branch_cycle_ntimes(n_pgibbs_particles::Integer, n_cycles::Integer)
    if n_cycles == 0
        tr -> tr
    elseif n_cycles == 1
        return tr -> resimulate_branch_cycle(tr, n_pgibbs_particles)
    else
        return tr -> resimulate_branch_cycle_ntimes(n_pgibbs_particles, n_cycles - 1)(resimulate_branch_cycle(tr, n_pgibbs_particles))
    end
end

do_smc_inference(groundtruth_tr::Gen.Trace, nparticles, n_pgibbs_particles, ncycles_per_step) =
    dynamic_model_smc(model,
        get_dynamic_model_obs(groundtruth_tr),
        cm -> (cm[:number => :number => :val],),
        initial_proposal, step_proposal, nparticles;
        rejuvenate=resimulate_branch_cycle_ntimes(n_pgibbs_particles, ncycles_per_step)
    );

### Labeling util functions: ###

get_title(nums, n_smc_particles, n_pg_particles, n_rejuv_sweeps_per_iter) = """
P[# in set | $nums] | maxdepth=$(MAXDEPTH())
Resample-Move w/ Branch Resimulation PGibbs Rejuvenation.
$n_smc_particles SMC Particles. $n_pg_particles PG particles.  $n_rejuv_sweeps_per_iter rejuvenation sweeps per timestep.
$(ngf_str())
"""

ngf_str() = use_ngf() ? """
NeuralGen-Fast with MaxRate=$(ProbEstimates.MaxRate())Hz, As.Size=$(ProbEstimates.AssemblySize()), E[Latency]=$(ProbEstimates.Latency())ms.
$(ProbEstimates.UseLowPrecisionMultiply() ? "Resampling via low-precision single-line-compression multiplication." : "Resampling via auto-normalized multiplication (\"neural floating point\").")
""" : "Vanilla Gen."

### Run + make figures
filename_for_smc_run(nums, n_particles, n_pgibbs_particles, n_rejuv_sweeps) = reduce(*, ["$(n)_" for n in nums]) * "__$(n_particles)smc_$(n_pgibbs_particles)pg_$(n_rejuv_sweeps)rejuv" * ".png"
filename_for_enumeration_run(nums) = reduce(*, ["$(n)_" for n in nums]) * "__enumeration" * ".png"
obs_choicemap(nums) = choicemap(
    (:init => :obs => :number => :number => :val, nums[1]),
    (
        (:steps => t => :obs => :number => :number => :val, num)
        for (t, num) in enumerate(nums[2:end])
    )...
)
trace_with_nums(nums) = generate(model, (length(nums) - 1,), obs_choicemap(nums))[1]

function do_enumeration_save_fig(nums)
    membership_probs = get_number_membership_probs(obs_choicemap(nums), length(nums) - 1)
    f = visualize(nums, membership_probs; title="P[# in set | $nums] | maxdepth=$(MAXDEPTH()) | Enumeration Results\n.\n.\n.\n.")
    save(filename_for_enumeration_run(nums), f)
end

function do_smc_inference_on_nums_and_save_fig(
    nums, n_particles, n_pgibbs_particles, n_rejuv_sweeps=1;
    title=get_title(nums, n_particles, n_pgibbs_particles, n_rejuv_sweeps),
    resolution=(400, 200)
)
    (_, weighted_trs) = do_smc_inference(trace_with_nums(nums), n_particles, n_pgibbs_particles, n_rejuv_sweeps)
    end_weighted_traces = [(tr, exp(wt)) for (tr, wt) in last(weighted_trs)]

    f = visualize_weighted_traces(
        end_weighted_traces; title, resolution
    )
    
    save(filename_for_smc_run(nums, n_particles, n_pgibbs_particles, n_rejuv_sweeps), f)
end

### Spiketrain figures:

function get_num_possibilities(pvec, n)
    possibilities = findall([i for (i, p) in enumerate(pvec) if p > 0])
    if n == first(possibilities)
        return possibilities[1:3]
    elseif n == possibilities[end]
        return possibilities[end-2:end]
    else
        idx = findfirst([i == n for i in possibilities])
        return possibilities[(idx-1):(idx+1)]
    end
end
function n1s(tr)
    ch = get_submap(get_choices(tr), :init => :latents => :tree => :teminal)
    n1 = ch[:n1 => :val]
    n1vec = n1_pvec(ch[:typ => :val])
    return get_num_possibilities(n1vec, n1)
end
function n2s(tr)
    ch = get_submap(get_choices(tr), :init => :latents => :tree => :teminal)
    n1 = ch[:n1 => :val]
    n2 = ch[:n2 => :val]
    n2vec = n2_pvec(ch[:typ => :val], n1)
    return get_num_possibilities(n2vec, n2)
end

function make_spiketrain_fig(tr, neurons_to_show_indices=1:3; kwargs...)
    nest_all_at = :init => :latents => :tree

    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
    assess_sampling_tree = Dict(
        :is_terminal => [],
        (:terminal => :typ) => [],
        (:terminal => :n1) => [(:terminal => :typ)],
        (:terminal => :n2) => [(:terminal => :typ), (:terminal => :n2)]
    )
    propose_sampling_tree = assess_sampling_tree
    propose_addr_topological_order = [:is_terminal, :terminal => :typ, :terminal => :n1, :terminal => :n2]
    
    doms = [
        [true, false],
        [:prime, :multiple_of, :interval],
        n1s(tr), n2s(tr)
    ]
    return ProbEstimates.Spiketrains.draw_spiketrain_group_fig(
        ProbEstimates.Spiketrains.value_neuron_scores_groups(keys(doms), values(doms), neurons_to_show_indices), 
        tr, (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order);
        nest_all_at, kwargs...
    )
end

### Code to do a bunch of runs:
# specs = list of `(nums, n_particles, n_pgibbs_particles, n_rejuvenation_sweeps_per_timestep)`
function do_smc_runs(specs; override_title_to=nothing)
    for (i, spec) in enumerate(specs)
        @info "On spec $i / $(length(specs))."
        try_run() = try
            if isnothing(override_title_to)
                do_smc_inference_on_nums_and_save_fig(spec...)
            else
                do_smc_inference_on_nums_and_save_fig(spec...; title=override_title_to)
            end
            true
        catch e
            @error "$e"
            false
        end
        while !try_run();
            println("attempting again...")
        end
    end
end

### Script to actually do a particular run:

early_nums = [30, 31, 33, 24, 21, 36, 39]
late_nums = [30, 33, 24, 21, 36, 31, 39]
# numss = [
#     early_nums,
#     late_nums,
#     [33, 24, 21, 36, 31, 39, 30],
#     [24, 21, 36, 31, 39, 30, 33],
#     [21, 36, 31, 39, 30, 33, 21],
#     [36, 31, 39, 30, 33, 21, 21]
# ]

numss = [
    late_nums[1:2],
    late_nums[1:4],
    late_nums[1:6]
]
specs = [
    (nums, 100, 2, 2) for nums in numss
]
do_smc_runs(specs; override_title_to="")

# specs = Iterators.flatten(
#     (
#         (late_nums, 100, n_pg, 0),
#         (late_nums, 100, n_pg, 2),
#         (late_nums, 1,   n_pg, 10)
#     )
#     for n_pg in (2, 10, 100)
# ) |> collect
# do_smc_runs(specs)
# do_enumeration_save_fig(late_nums)

### Spiketrain Figure:
# (_, weighted_trs) = do_smc_inference(trace_with_nums(nums), n_particles, n_pgibbs_particles, n_rejuv_sweeps)
# f = make_spiketrain_fig(last(unweighted_trs)[1]; resolution=(600, 450), title="Dynamically Weighted Spike Code from Inference")