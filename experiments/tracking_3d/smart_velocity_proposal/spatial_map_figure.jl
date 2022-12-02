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
(((f, t), (times, group_labels, colors, n_hidden_lines)), hidden_line_specs) = make_anim_spiketrain_fig_and_get_all_L4_with_all_coords(
    traces_at_each_time[3:7], logweights_at_each_time[3:7], 1:100;
    figure_title="Spikes from SMC Neurons for 3D Tracking",
    resolution=(750, 1000), return_metadata=true,
    first_label_length=170
); # We aren't going to use this Makie figure.  We're gonna make a new one.

get_xy(r, ϕ, θ) = (r * cos(ϕ) * cos(θ), r * cos(ϕ) * sin(θ))
get_xy(r, θ) = get_xy(r, 0, θ)
domain(vname) = get(Dict("x" => Xs(), "y" => Ys(), "θ" => θs(), "r" => Rs()), String(vname), nothing)
vname_to_label(vname) = 
    if vname == "θ"
        "Q[true_θ]"
    else
        "Q[$vname]"
    end
function get_spatial_map_figure(times, hidden_line_specs; W=10)
    f = Figure(resolution=(500, 1000))
    t = Observable(0)
    aloax = Axis(f[1, 1], title="Activity in L4 XYZ Samplers (2D projection)")
    aloax.xgridvisible = false; aloax.ygridvisible = false;
    egoax = Axis(f[2, 1], title="Activity in L4 Rϕθ Samplers (2D projection)")

    # returns a number in [0, 1]
    all_times = sort(collect(Iterators.flatten(times[1:n_hidden_lines])))
    labels = [group.label_spec.text for group in hidden_line_specs]

    lengths = [length(group.line_specs) for group in hidden_line_specs]
    starting_indices = cumsum([1, lengths...])
    var_idx(vname) = findfirst(labels .== vname_to_label(vname))::Int
    val_idx(vname, val) = try findfirst(domain(vname) .== val)::Int; catch e; println("$vname, $val"); throw(e); end;
    times_for_var(vname, val) = times[starting_indices[var_idx(vname)] + val_idx(vname, val)]
    count_in_window(vname, val, t, w) = count(x -> t - w ≤ x ≤ t, times_for_var(vname, val))
    get_var_activity(vname, val, t, w) = min(length(domain(vname)) * 10 * count_in_window(vname, val, t, w) / length(all_times), 1.)

    for x in Xs()
        lines!(aloax,
            [Point2(x, minimum(Ys())), Point2(x, maximum(Ys()))],
            color=@lift(colormap("reds")[Int(floor(99 * get_var_activity("x", x, $t, W))) + 1])
        )
    end
    for y in Ys()
        lines!(aloax,
            [Point2(minimum(Xs()), y), Point2(maximum(Xs()), y)],
            color=@lift(colormap("reds")[Int(floor(99 * get_var_activity("y", y, $t, W))) + 1])
        )
    end

    for r in Rs()
        lines!(egoax,
            [Point2(get_xy(r, θ)) for θ in θs()],
            color=@lift(colormap("reds")[Int(floor(99 * get_var_activity("r", r, $t, W))) + 1])
        )
    end
    for θ in θs()
        lines!(egoax,
            [Point2(get_xy(0, θ)), Point2(get_xy(maximum(Rs()), θ))],
            color=@lift(colormap("reds")[Int(floor(99 * get_var_activity("θ", θ, $t, W))) + 1])
        )
    end

    xlims!(egoax, (minimum(Xs()), maximum(Xs())))
    ylims!(egoax, (minimum(Ys()), maximum(Ys())))
    xlims!(aloax, (minimum(Xs()), maximum(Xs())))
    ylims!(aloax, (minimum(Ys()), maximum(Ys())))

    return (f, t)
end

(f, t) = get_spatial_map_figure(times, hidden_line_specs; W=10); t[] = 10; f

record(f, "spatial_map.gif", 10:90;
        framerate = 10) do tval
    t[] = tval
end