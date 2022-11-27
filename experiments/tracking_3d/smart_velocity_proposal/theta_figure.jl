using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
import DynamicModels
using ProbEstimates

const SpTrs = ProbEstimates.Spiketrains

unzip(list) = ([x for (x, y) in list], [y for (x, y) in list])

includet("../model.jl")
includet("../ab_viz.jl")
include("deferred_inference.jl")
includet("spiketrain_fig.jl")

### Set hyperparameters ###
ProbEstimates.MinProb() = 0.01
println("MinProb() = $(MinProb())")
# ProbEstimates.use_perfect_weights!()
ProbEstimates.use_noisy_weights!()
ProbEstimates.AssemblySize() = 2000
ProbEstimates.Latency() = 15
ProbEstimates.UseLowPrecisionMultiply() = false
ProbEstimates.MultAssemblySize() = 600
ProbEstimates.AutonormalizeRepeaterAssemblysize() = 100
ProbEstimates.TimerExpectedT() = 25
ProbEstimates.TimerAssemblySize() = 20
ProbEstimates.AutonormalizeCountThreshold() = 5
ProbEstimates.MaxRate() = 0.2 # KHz

model = @DynamicModel(initial_model, step_model, obs_model, 9)
initial_proposal_compiled = @compile_initial_proposal(initial_proposal, 2)
step_proposal_compiled = @compile_step_proposal(step_proposal, 9, 2)
two_timestep_proposal_dumb = @compile_2timestep_proposal(initial_proposal, step_proposal, 9, 2)

@load_generated_functions()

NSTEPS = 8
NPARTICLES = 16
cmap = get_selected(make_deterministic_trace(), select(:init, :steps => 1, :steps => 2, :steps => 3, :steps => 4))
tr, w = generate(model, (NSTEPS,), cmap)
observations = get_dynamic_model_obs(tr);

### Run inference, and record the resulting traces.
### Do this multiple times in case some runs come back with -Inf weights.
final_particle_set = []
resampling_indices = []
unweighted_traces_at_each_step_vector = []
weighted_traces_vec = []
while isempty(final_particle_set)
    for i in 1:5
        (unweighted_traces_at_each_step, weighted_traces, resample_inds) = deferred_dynamic_model_smc(
            model, (observations[1], observations[2][1:NSTEPS]),
            ch -> (ch[:obs_ϕ => :val], ch[:obs_θ => :val]),
            two_timestep_proposal_dumb,
            # propose_first_two_timesteps_smart,
            step_proposal_compiled,
            NPARTICLES, # n particles
            ess_threshold=NPARTICLES,
            get_resampling_indices=true
        );

        weights = map(x -> x[2], weighted_traces[end])
        particles = map(x -> x[1], weighted_traces[end])
        pvec = normalize(exp.(weights .- logsumexp(weights)))
        if !isprobvec(pvec)
            continue
        else
            sample = Gen.categorical(pvec)
            push!(final_particle_set, particles[sample])

            push!(resampling_indices, resample_inds)
            push!(weighted_traces_vec, weighted_traces)
        end
    end
end

weighted_traces = first(weighted_traces_vec)
ancestor_indices = first(resampling_indices)
logweights_at_each_time = [[logweight for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_traces ]
traces_at_each_time = [[trace for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_traces ]

### Spiketrain visualization ###
function weight_autonorm_groups(particle_indices)
    weight_groups = [
        SpTrs.LabeledMultiParticleLineGroup(
            SpTrs.FixedText("Particle $part_idx normalized weight"),
            [SpTrs.NormalizedWeight(part_idx, SpTrs.CountAssembly())]
        )
        for part_idx in particle_indices
    ]

    autonorm_group = SpTrs.LabeledMultiParticleLineGroup(SpTrs.FixedText("≈-log(P[d])"), [SpTrs.LogNormalization(SpTrs.CountAssembly())])

    return vcat(weight_groups, [autonorm_group])
end

function add_lines_at_starttime!(lines, lines_to_add, starttime)
    if isempty(lines)
        for line in lines_to_add
            push!(lines, line .+ starttime)
        end
    else
        @assert length(lines) == length(lines_to_add)
        for i in eachindex(lines)
            append!(lines[i], lines_to_add[i] .+ starttime)
        end
    end
end

function getspikes(
    trs, log_trace_weights, ancestor_indices, n_particles_per_timestep,
    (prop_sample_tree, assess_sample_tree, prop_addr_top_order, addr_to_domain);
    time_to_nesting_addr=ProbEstimates.Spiketrains.default_t_to_nesting_address,
    # Default: ~1.1 for weight readout, .1 for autonormalization excitatory spikes, 1 for weight readout
    timestep_length_to_latency_ratio=2.5,
    ms_per_step=25,
    kwargs...
)
    ## Get lines over multiple timesteps
    ms_per_timestep = timestep_length_to_latency_ratio * ProbEstimates.Latency()
    particle_lines = []
    resample_lines = []
    starttime = -ms_per_timestep
    for (t_plus_1, (trs_at_step, logweights_at_step)) in enumerate(zip(trs, log_trace_weights))

        # This is for the case where particles are processed sequentially in batches
        N = length(trs_at_step)
        for first_idx=1:n_particles_per_timestep:N
            trs = trs_at_step[first_idx:min(N, first_idx + n_particles_per_timestep)]
            logweights = logweights_at_step[first_idx:min(N, first_idx + n_particles_per_timestep)]

            starttime += ms_per_step

            if any(logweights .> -Inf)
                lines_now = ProbEstimates.Spiketrains.get_lines_for_multiparticle_spec_groups(
                    weight_autonorm_groups([1]), trs, logweights,
                    (prop_sample_tree, assess_sample_tree, prop_addr_top_order, addr_to_domain);
                    nest_all_at=time_to_nesting_addr(t_plus_1 - 1)
                )
            else
                lines_now = [[], []]
            end

            add_lines_at_starttime!(particle_lines, lines_now, starttime)
        end

        @assert !isempty(particle_lines)
        ready_time = maximum([lines[end] - starttime for lines in particle_lines])
        println("ready_time = $ready_time")
        resampling_lines_now = SpTrs.resampler_groups(ancestor_indices[t_plus_1], ready_time, 40)
        num = sum(length.(resampling_lines_now))
        println("Num resampling spikes: $num")
        add_lines_at_starttime!(resample_lines, resampling_lines_now, starttime)
    end

    return (particle_lines, resample_lines)
end

(particle_lines, resample_lines) =
    getspikes(
        traces_at_each_time, logweights_at_each_time, ancestor_indices, 4,
        get_trees_etc(traces_at_each_time, logweights_at_each_time);
        timestep_length_to_latency_ratio=5/3
    )

all_times = sort(collect(Iterators.flatten(Iterators.flatten((particle_lines, resample_lines)))))

function draw_wave(all_times)
    count_in_window(t, w) = count(x -> t - w ≤ x ≤ t, all_times)

    f = Figure()
    AVG = 10
    ax = Axis(f[1, 1], title="Overall Basal Ganglia activity ($AVG ms average)")

    xs = AVG:0.05:750
    hideydecorations!(ax)
    # hidexdecorations!(ax)
    lines!(ax, xs, map(x -> count_in_window(x, AVG), xs), color=:black, linewidth=6)
    
    f
end
draw_wave(all_times)

# maximum.([wave[((x - AVG)/.05 |> Int):((x + 25))] for x=25:25:600])

# count_in_window(t, w) = count(x -> t - w ≤ x ≤ t, all_times)
# AVG = 10
# STEP = 1
# xs = AVG:STEP:750
# raw_wave = [count_in_window(x, AVG) for x in xs]




# function middle_n(arr, n)
#     pad = length(arr) - n
#     st = Int(floor(pad/2) + 1)
#     nd = st + n - 1
#     return arr[st:nd]
# end
# gauss = DSP.Windows.gaussian(10, 1)
# filtered = DSP.conv(xs, gauss)
# lines(xs, middle_n(filtered, length(xs)))


# responsetype = Lowpass(150; fs=1e3)
# designmethod = FIRWindow(hanning(128; zerophase=false))
# #designmethod = FIRWindow(hanning(64; padding=30, zerophase=false))
# #designmethod = FIRWindow(rect(128; zerophase=false))
# #designmethod = Butterworth(4)
# wave_low = filt(digitalfilter(responsetype, designmethod), raw_wave)
