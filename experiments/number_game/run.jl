using Gen, ProbEstimates, DynamicModels

ProbEstimates.MinProb() = 1/100
ProbEstimates.Latency() = 100.
ProbEstimates.AssemblySize() = 1000
ProbEstimates.MaxRate() = 0.1
ProbEstimates.MultAssemblySize() = 500
ProbEstimates.AutonormalizeRepeaterAssemblysize() = 10
ProbEstimates.UseLowPrecisionMultiply() = false
ProbEstimates.AutonormalizeCountThreshold() = 5

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

function regenerate_branch_cycle_ntimes(n_pgibbs_particles::Integer, n_cycles::Integer; proposal=repropose_tree_branch_data_driven) #repropose_tree_branch_data_driven) #resample_tree_branch)
    if n_cycles == 0
        tr -> tr
    elseif n_cycles == 1
        return tr -> rejuvenate_branch_cycle(tr, n_pgibbs_particles; proposal)
    else
        return tr -> regenerate_branch_cycle_ntimes(n_pgibbs_particles, n_cycles - 1)(rejuvenate_branch_cycle(tr, n_pgibbs_particles))
    end
end

do_smc_inference(groundtruth_tr::Gen.Trace, nparticles, n_pgibbs_particles, ncycles_per_step) =
    dynamic_model_smc(model,
        get_dynamic_model_obs(groundtruth_tr),
        cm -> (cm[:number => :number => :val],),
        initial_proposal, step_proposal, nparticles;
        rejuvenate=regenerate_branch_cycle_ntimes(n_pgibbs_particles, ncycles_per_step)
    );

### Labeling util functions: ###

get_title(nums, n_smc_particles, n_pg_particles, n_rejuv_sweeps_per_iter) = """
P[# in set | $nums] | maxdepth=$(MAXDEPTH())
Resample-Move w/ Branch Re-Proposal PGibbs Rejuvenation.
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
nums_to_obs_cm(nums) = choicemap(
    (:init => :obs => :number => :number => :val, nums[1]),
    (
        (:steps => t => :obs => :number => :number => :val, num)
        for (t, num) in enumerate(nums[2:end])
    )...
)
trace_with_nums(nums) = generate(model, (length(nums) - 1,), nums_to_obs_cm(nums))[1]

function do_enumeration_save_fig(nums;
    title="P[number in set | maxdepth=$(MAXDEPTH())] : Enumeration Results\n.\n.\n.\n.",
    fontsize
)
    membership_probs = get_number_membership_probs(nums_to_obs_cm(nums), length(nums) - 1)
    f = visualize(nums, membership_probs; title, fontsize)
    save(filename_for_enumeration_run(nums), f)
end

function do_smc_inference_on_nums_and_save_fig(
    nums, n_particles, n_pgibbs_particles, n_rejuv_sweeps=1;
    title=get_title(nums, n_particles, n_pgibbs_particles, n_rejuv_sweeps),
    resolution=(400, 200), fontsize=20
)
    (_, weighted_trs) = do_smc_inference(trace_with_nums(nums), n_particles, n_pgibbs_particles, n_rejuv_sweeps)
    end_weighted_traces = [(tr, exp(wt)) for (tr, wt) in last(weighted_trs)]

    f = visualize_weighted_traces(
        end_weighted_traces; title, resolution, fontsize
    )
    
    filename = filename_for_smc_run(nums, n_particles, n_pgibbs_particles, n_rejuv_sweeps)
    println("saving at $filename")
    save(filename, f)
end

### Spiketrain figures:

function get_num_possibilities(pvec, n)
    possibilities = [i for (i, p) in enumerate(pvec) if p > 0]
    if length(possibilities) < 3
        return possibilities
    elseif n == first(possibilities)
        return possibilities[1:3]
    elseif n == possibilities[end]
        return possibilities[end-2:end]
    else
        idx = findfirst([i == n for i in possibilities])
        return possibilities[(idx-1):(idx+1)]
    end
end
function n1s(tr)
    ch = get_submap(get_choices(tr), :init => :latents => :tree => :terminal)
    n1 = ch[:n1 => :val]
    n1vec = n1_pvec(ch[:typ => :val])
    return get_num_possibilities(n1vec, n1)
end
function n2s(tr)
    ch = get_submap(get_choices(tr), :init => :latents => :tree => :terminal)
    n1 = ch[:n1 => :val]
    n2 = ch[:n2 => :val]
    n2vec = n2_pvec(ch[:typ => :val], n1)
    return get_num_possibilities(n2vec, n2)
end

### Code to do a bunch of runs:
# specs = list of `(nums, n_particles, n_pgibbs_particles, n_rejuvenation_sweeps_per_timestep)`
function do_smc_runs(specs; titles=[nothing for spec in specs], resolution=(200, 400), fontsize=20)
    for (i, (spec, title)) in enumerate(zip(specs, titles))
        @info "On spec $i / $(length(specs))."
        try_run() = try
            if isnothing(title)
                do_smc_inference_on_nums_and_save_fig(spec...; resolution, fontsize)
            else
                do_smc_inference_on_nums_and_save_fig(spec...; title, resolution, fontsize)
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

# numss = [
#     late_nums[1:2],
#     late_nums[1:4],
#     late_nums[1:6]
# ]
# specs = [
#     (nums, 100, 2, 2) for nums in numss
# ]
specs = [(late_nums, 20, 2, 4)]
do_smc_runs(specs)

# specs = [
#     (late_nums, 100, 100, 2),
#     (late_nums, 100, 2, 0)
# ]
# titles = ["Inferred P[number in set ; observed numbers]", "Inferred P[number in set ; observed numbers]"]
# do_smc_runs(specs; titles)

do_enumeration_save_fig(late_nums; title="Exact P[number in set ; observed numbers]", fontsize=20)

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

function make_spiketrain_fig_numgame(trs, logweights, neurons_to_show_indices=1:10; kwargs...)
    nest_all_at = :init => :latents => :tree

    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
    assess_sampling_tree = Dict(
        :is_terminal => [],
        (:terminal => :typ) => [],
        (:terminal => :n1) => [(:terminal => :typ)],
        (:terminal => :n2) => [(:terminal => :typ), (:terminal => :n1)]
    )
    propose_sampling_tree = assess_sampling_tree
    propose_addr_topological_order = [:is_terminal, :terminal => :typ, :terminal => :n1, :terminal => :n2]
    
    addr_to_name = Dict(
        :is_terminal => :s,
        (:terminal => :typ) => :τ,
        (:terminal => :n1) => :n1,
        (:terminal => :n2) => :n2,
    )

    addr_to_domain = Dict(
        :is_terminal => [true, false],
        (:terminal => :typ) => [:prime, :multiple_of, :interval],
        (:terminal => :n1) => [i for i=1:100 if 1 ≤ i ≤ 10 || i % 5 == 0],
        (:terminal => :n2) => 1:100,

    )

    doms = [
        [true, false],
        [:prime, :multiple_of, :interval],
        [i for i=1:100 if 2 ≤ i ≤ 10 || i % 5 == 0], 1:100
    ]

    max_weight_idx_at_each_time = [
        findmax(arr)[2] for arr in logweights
    ]
    println("indices = $max_weight_idx_at_each_time")

    function time_to_nesting_addr(t)
        # t == 0 ? (:init => :latents => :tree) : (:steps => t => :latents => :tree)
        return :init => :latents => :tree # PGibbs ALWAYS operates on the initial timestep's choices
    end

    variables_vals_to_show_p_dists_for = [
        (:is_terminal, [true, false]),
        [:terminal => :typ, [:prime, :multiple_of, :interval]]
    ]
    variables_vals_to_show_q_dists_for = [
        (:is_terminal, [true, false]),
        [:terminal => :typ, [:prime, :multiple_of, :interval]]
    ]
    function val_to_label(val)
        if val === true
            "T"
        elseif val === false
            "F"
        elseif val === :prime
            ":p"
        elseif val === :multiple_of
            ":m"
        elseif val === :interval
            ":i"
        else
            error("unexpected value $val")
        end
    end

    return ProbEstimates.Spiketrains.draw_multiparticle_multistep_spiketrain_group_fig(
        ProbEstimates.Spiketrains.value_neuron_scores_dists_weight_autonorm_groups(
            propose_addr_topological_order, doms, max_weight_idx_at_each_time[1], max_weight_idx_at_each_time,
            variables_vals_to_show_p_dists_for, variables_vals_to_show_p_dists_for,
            neurons_to_show_indices; addr_to_name=(a -> addr_to_name[a]), val_to_label,
            mult_neurons_to_show_indices=1:50
        ),
        trs, logweights,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order, addr_to_domain);
        timestep_length_to_latency_ratio=8/3,
        figure_title="Spikes from PGibbs Neurons for Concept Learning",
        time_to_nesting_addr,
        resolution=(600, 600),
        addr_to_name=(a -> addr_to_name[a]),
        kwargs...
    )
    # return ProbEstimates.Spiketrains.draw_spiketrain_group_fig(
    #     ProbEstimates.Spiketrains.value_neuron_scores_groups_noind(
    #         propose_addr_topological_order, doms, neurons_to_show_indices,
    #         addr_to_name=(a -> addr_to_name1[a])
    #     ), 
    #     tr, (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order);
    #     nest_all_at, kwargs...
    # )
end

# Do with 1 rejuvenation sweep.  TODO: extend spiketrain-making code so it supports doing multiple rejuvenation sweeps
(unweighted_trs, weighted_trs) = do_smc_inference(trace_with_nums(late_nums), 50, 2, 1);
logweights_at_each_time = [[logweight for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_trs ]
traces_at_each_time = [[trace for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_trs ]
get_f() = make_spiketrain_fig_numgame(traces_at_each_time[2:4], logweights_at_each_time[2:4], 1:20)
f = get_f()




# function get_fig()
#     for i=1:100
#         try
#             return get_f(i)
#         catch e
#         end
#     end
# end
# ProbEstimates.Spiketrains.SpiketrainViz.save("concept_learning.png", get_fig())
