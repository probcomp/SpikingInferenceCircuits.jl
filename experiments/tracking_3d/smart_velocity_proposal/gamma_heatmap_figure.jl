using Base: Int64
using DynamicModels: @DynamicModel, @compile_initial_proposal, @compile_step_proposal, get_dynamic_model_obs, dynamic_model_smc
import DynamicModels
using ProbEstimates

unzip(list) = ([x for (x, y) in list], [y for (x, y) in list])

includet("../model.jl")
includet("../ab_viz.jl")
include("deferred_inference.jl")

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
NPARTICLES = 10
cmap = get_selected(make_deterministic_trace(), select(:init, :steps => 1, :steps => 2, :steps => 3, :steps => 4))
tr, w = generate(model, (NSTEPS,), cmap)
observations = get_dynamic_model_obs(tr);

### Run inference, and record the resulting traces.
### Do this multiple times in case some runs come back with -Inf weights.
final_particle_set = []
unweighted_traces_at_each_step_vector = []
weighted_traces_vec = []
for i in 1:5
    # try
        (unweighted_traces_at_each_step, weighted_traces) = deferred_dynamic_model_smc(
            model, (observations[1], observations[2][1:NSTEPS]),
            ch -> (ch[:obs_ϕ => :val], ch[:obs_θ => :val]),
            two_timestep_proposal_dumb,
            # propose_first_two_timesteps_smart,
            step_proposal_compiled,
            NPARTICLES, # n particles
            ess_threshold=NPARTICLES
        );

        weights = map(x -> x[2], weighted_traces[end])
        particles = map(x -> x[1], weighted_traces[end])
        pvec = normalize(exp.(weights .- logsumexp(weights)))
        if !isprobvec(pvec)
            continue
        else
            sample = Gen.categorical(pvec)
            push!(final_particle_set, particles[sample])

            push!(weighted_traces_vec, weighted_traces)
        end

    # catch
    #     continue
    # end
end
length(final_particle_set)

# animate_pf_results(final_particle_set, tr, true)
# animate_pf_results(final_particle_set, tr, false)
# render_static_trajectories(final_particle_set, tr, true)
# render_static_trajectories(final_particle_set, tr, false)
# final_scores = [get_score(t) for t in final_particle_set]
# final_probs = normalize(exp.(final_scores .- logsumexp(final_scores)))
# render_obs_from_particles(final_particle_set, 100; do_obs=false);


### Spiketrain visualization ###
includet("spiketrain_fig.jl") ### File with code for this specific spiketrain visualization

### Get the particle weights and traces at each timestep
weighted_traces = first(weighted_traces_vec)
logweights_at_each_time = [[logweight for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_traces ]
traces_at_each_time = [[trace for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_traces ]

### Make the Makie figure with the animation.
### `t` controls which window of time is shown on screen.
(((f, t), (times, group_labels, colors, n_hidden_lines)), hidden_line_specs) = make_anim_spiketrain_fig_and_get_all_L4(
    traces_at_each_time[3:7], logweights_at_each_time[3:7], 1:100;
    figure_title="Spikes from SMC Neurons for 3D Tracking",
    resolution=(750, 1000), return_metadata=true,
    first_label_length=170
); t[] = 45; GLMakie.activate!(); f

function augment_figure_with_wave!(f, times)
    all_times = sort(collect(Iterators.flatten(times[1:n_hidden_lines])))
    count_in_window(t, w) = count(x -> t - w ≤ x ≤ t, all_times)

    xs = 10:0.05:150
    ax = Axis(f[3, 1], title="Overall L4 activity (10ms average)")
    hideydecorations!(ax)
    hidexdecorations!(ax)
    lines!(ax, xs, map(x -> count_in_window(x, 10), xs), color=:black, linewidth=6)
    onany(t) do t # update the limits at the given times
        xlims!(ax, (t[], t[] + 75))
    end
    t[] = 0
    rowsize!(f.layout, 2, Relative(.3))
    f
end
function augment_figure_with_heatmap!(f, hidden_line_specs, times)
    labels = [group.label_spec.text for group in hidden_line_specs]
    lengths = [length(group.line_specs) for group in hidden_line_specs]
    starting_indices = cumsum([1, lengths...])
    all_times_in_group(group_idx) = 
        let st = starting_indices[group_idx],
            range = st:(st+lengths[group_idx])
            sort(collect(Iterators.flatten(times[range])))
    end
    count_in_window(t, w, group_idx) = count(x -> t - w ≤ x ≤ t, all_times_in_group(group_idx))
    
    total_count(group_idx) = length(all_times_in_group(group_idx))
    fraction_in_window(t, w, group_idx) = count_in_window(t, w, group_idx) / total_count(group_idx)

    xs = 10:.1:150
    ax = Axis(f[2, 1], title="Local L4 activity (10ms average, rescaled locally)")
    ax.yreversed = true
    hideydecorations!(ax)
    hidexdecorations!(ax)
    heatmap!(ax, xs, 1:length(labels), [
        fraction_in_window(x, 10, y)
        for x in xs, y=1:length(labels)
    ], colormap=:greys)
    onany(t) do t # update the limits at the given times
        xlims!(ax, (t[], t[] + 75))
    end
    t[] = 0
    rowsize!(f.layout, 2, Relative(.3))

    ProbEstimates.Spiketrains.SpiketrainViz.draw_group_labels!(f, ax, [(l, 1) for l in reverse(labels)], 0, [:black for _ in labels])

    f
end
augment_figure_with_heatmap!(f, hidden_line_specs, times)
augment_figure_with_wave!(f, times)

f

#=
times[i] = the ith spike line in the figure (top to bottom)
group_labels[1][j] = (label, the number of lines in the jth group top to bottom)
=#

### Animate time passing.
record(f, "spiketrain_gamma_heatmap_10msavg_3.gif", 10:0.5:90;
        framerate = 10) do tval
    t[] = tval
end

















# plot_full_choicemap(final_particle_set)


# unweighted_traces_at_each_step looks like
# [
    # [particle1trace, particle2trace, ...] # for timestep 1
    # [particle1trace, particle2trace, ...] # for timestep 2
    # [particle1trace, particle2trace, ...] # for timestep 3
# ]
# where the traces are the traces we have after resampling

# by a "trace for timestep T", I mean a trace which has choices
# for every timestep up to and including T

#tr_init = simulate(model, (0,))
#proposed_choices, _ = propose(step_proposal, (tr_init, 0.0, 0.0))
#[propose(step_proposal, (tr_init, 0.0, 0.0))[1][:steps => 1 => :latents => :moving_in_depthₜ] for i in 1:200]

