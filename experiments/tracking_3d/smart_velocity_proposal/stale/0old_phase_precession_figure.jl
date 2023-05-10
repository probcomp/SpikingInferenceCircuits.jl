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

NSTEPS = 14
NPARTICLES = 3
cmap = get_selected(make_deterministic_trace(), select(:init, (:steps => i for i=1:NSTEPS)...))
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
((_, (times, group_labels, colors, n_hidden_lines)), hidden_line_specs) = get_all_L4_with_all_coords(
    traces_at_each_time[2:end], logweights_at_each_time[2:end], 1:100;
    figure_title="Spikes from SMC Neurons for 3D Tracking",
    resolution=(750, 1000), return_metadata=true,
    first_label_length=170
); # We aren't going to use this Makie figure.  We're gonna make a new one.

function augment_figure_with_wave!(f, times)
    all_times = sort(collect(Iterators.flatten(times[1:n_hidden_lines])))
    count_in_window(t, w) = count(x -> t - w ≤ x ≤ t, all_times)

    xs = 10:0.05:150
    ax = Axis(f[2, 1], title="Overall L4 activity (10ms average)")
    hideydecorations!(ax)
    hidexdecorations!(ax)
    lines!(ax, xs, map(x -> count_in_window(x, 10), xs), color=:black, linewidth=6)
    onany(t) do t # update the limits at the given times
        xlims!(ax, (t[], t[] + 25))
    end
    t[] = 0
    rowsize!(f.layout, 2, Relative(.3))
    f
end

get_xy(r, ϕ, θ) = (r * cos(ϕ) * cos(θ), r * cos(ϕ) * sin(θ))
get_xy(r, θ) = get_xy(r, 0, θ)
domain(vname) = get(Dict("x" => Xs(), "y" => Ys(), "true_θ" => θs(), "r" => Rs()), String(vname), nothing)
vname_to_label(vname) = 
    if vname == "θ"
        "P[true_θ]"
    else
        "P[$vname]"
    end
Γ_period = 25
@dist uniform_from_list(list) = list[uniform_discrete(1, length(list))]

function get_clusters(list, threshold=1.)
    intervals = [list[i+1] - list[i] for i=1:(length(list)-1)]
    indices_of_real_gaps = vcat(findall(intervals .> threshold), [length(intervals)])
    cluster_start_points = [1, (idx + 1 for idx in indices_of_real_gaps)...]
    return [
        list[i:g] for (i, g) in zip(cluster_start_points, indices_of_real_gaps)
    ]
end

function augment_figure_with_scatter!(f, hidden_line_specs, times, y1=-0.4, y2=0.4)
    ax = Axis(f[1, 1])

    labels = [group.label_spec.text for group in hidden_line_specs]
    lengths = [length(group.line_specs) for group in hidden_line_specs]
    starting_indices = cumsum([1, lengths...])
    var_idx(vname) = findfirst(labels .== vname_to_label(vname))::Int
    val_idx(vname, val) = try findfirst(domain(vname) .== val)::Int; catch e; println("$vname, $val"); throw(e); end;
    times_for_var(vname, val) = times[starting_indices[var_idx(vname)] + val_idx(vname, val)]

    true_positions = [observations[2][i][:obs_θ => :val] for i=1:NSTEPS]
    spiketimes_at_timestep(tstep, y) = let clusters = get_clusters(times_for_var("true_θ", y))
        tstep > length(clusters) ? [] : clusters[tstep]
    end
    #[ t for t in times_for_var("true_θ", y) if Γ_period * (tstep - 1) ≤ t ≤ Γ_period * tstep ]
    Γ_phase_deg(t) = 360 * (t / Γ_period - floor(t / Γ_period))
    spike_phases_at_timestep(tstep, y) = collect(sort(map(Γ_phase_deg, spiketimes_at_timestep(tstep, y))))
    pos_phase_pairs(tstep, y) = [ (true_positions[tstep], deg) for deg in spike_phases_at_timestep(tstep, y) ]
    all_pos_phase_pairs(y) = vcat((pos_phase_pairs(tstep, y) for tstep in 1:NSTEPS)...)
    subsample_100(vals) = vals #[uniform_from_list(vals) for _=1:100]
    scatter!(ax, map(Point2, all_pos_phase_pairs(y1)|>subsample_100), color=RGBA(colorant"red", 0.3))
    # scatter!(ax, map(Point2, all_pos_phase_pairs(y2)|>subsample_100), color=RGBA(colorant"blue", 0.3))
    scatter!(ax, map(((y, deg),) -> Point2(y, deg + 360), all_pos_phase_pairs(y1)|>subsample_100), color=RGBA(colorant"red", 0.3))
    # scatter!(ax, map(((y, deg),) -> Point2(y, deg + 360), all_pos_phase_pairs(y2)|>subsample_100), color=RGBA(colorant"blue", 0.3))

    ylims!(ax, (0, 720))
    ax.yticks = [0, 360, 720]
    ax.xlabel = "observed θ value"
    ax.ylabel = "phase relative to Γ at which spikes occurred in P[true_θ = $y1 | inferred xyz]"

    f
end

# augment_figure_with_wave!(f, times)
f = Figure()
augment_figure_with_scatter!(f, hidden_line_specs, times, -0.4, 0.2)


# (f, t) = get_spatial_map_figure(times, hidden_line_specs; W=10); t[] = 10; f

# record(f, "spatial_map.gif", 10:90;
#         framerate = 10) do tval
#     t[] = tval
# end