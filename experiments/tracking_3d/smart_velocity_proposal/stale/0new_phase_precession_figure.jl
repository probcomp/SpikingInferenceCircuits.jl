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
# cmap = get_selected(make_deterministic_trace(), select(:init, :steps => 1, :steps => 2, :steps => 3, :steps => 4))
# tr, w = generate(model, (NSTEPS,), cmap)
# observations = get_dynamic_model_obs(tr);
STARTθ = -0.6
observations = (
    choicemap((:obs_θ => :val, STARTθ - 0.4), (:obs_ϕ => :val, 0.3)),
    [
        choicemap((:obs_θ => :val, x), (:obs_ϕ => :val, 0.3))
        for x=(STARTθ - 0.2):0.2:(STARTθ + 1.2)
    ]
)

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
includet("spiketrain_fig.jl") 

function make_Qθ_spiketrain_fig_and_get_all_L4(trs_at_each_time, logweights_at_each_time, θ_values_to_show, neurons_to_show_indices=1:10; kwargs...)
    n_particles = length(first(trs_at_each_time))

    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()

    # Hard-code the dependency graph for the `P` and `Q` step generative functions.
    # (We could recover this with static compilation, but I haven't implemented that.)
    assess_sampling_tree = Dict(
        # :dx => [], :vyₜ => [], :vzₜ => [],
        # :xₜ => [:dx], :yₜ => [:vyₜ], :zₜ => [:vzₜ],
        :dx => [],
        :x => [:dx], :y => [], :z => [],
        :true_ϕ => [:x, :y, :z],
        :true_θ => [:x, :y, :z],
        :r => [:x, :y, :z, :true_θ, :true_ϕ]
        # :obs_θ => [:true_θ]
    )
    _propose_sampling_tree = [
        :true_θ => [], :true_ϕ => [],
        :r => [:true_θ, :true_ϕ],
        :x => [:true_θ, :true_ϕ, :r],
        :y => [:true_θ, :true_ϕ, :r],
        :z => [:true_θ, :true_ϕ, :r],
        :dx => [:x],
        # :vyₜ => [:true_θ, :true_ϕ, :rₜ],
        # :vzₜ => [:true_θ, :true_ϕ, :rₜ],
    ]

    propose_addr_topological_order = [p.first for p in _propose_sampling_tree]
    propose_sampling_tree = Dict(_propose_sampling_tree...)

    doms = latent_domains_for_viz(trs_at_each_time)

    # We use this to decide what particle to show spikes from.
    max_weight_idx_at_each_time = [
        findmax(arr)[2] for arr in logweights_at_each_time
    ]

    addr_to_domain = Dict(
        :true_θ => θs(), :true_ϕ => ϕs(), :r => Rs(), :x => Xs(), :y => Ys(), :z => Zs(),
        :dx => Vels(), :obs_θ => θs()
    )

    # We're going to show the P and Q distribution assemblies for these variables,
    # in this order. First element of tuple is a Bool; `true`<> P ; `false`<>Q.
    all_dists_to_show = [
        (false, :true_θ),
        # (true, :obs_θ),
        (false, :r),
        (false, :x),
        (true, :r),
        (false, :dx),
        (true, :true_θ),
        (true, :x),
        (true, :dx)
    ]
    # We not only need to know what variables to show the sampler neurons for,
    # but what values to show the sampler neurons for.
    # all_dists_vals_to_show = [
    #     (is_p, addr,
    #         get_value(get_choices(trs_at_each_time[1][max_weight_idx_at_each_time[1]]), :steps => 1 => :latents => addr)
    #     )
    #     for (is_p, addr) in all_dists_to_show
    # ]

    # dists_to_show = [(false, :true_θ)]
    dists_vals_to_show = [(false, :true_θ, θ) for θ in θ_values_to_show]

    full_specs = [
        ProbEstimates.Spiketrains.LabeledMultiParticleLineGroup(
            ProbEstimates.Spiketrains.FixedText("$(is_p ? "P" : "Q")[$addr]"),
            [
                ProbEstimates.Spiketrains.SubsidiarySingleParticleLineSpec(max_weight_idx_at_each_time[1], 
                    ProbEstimates.Spiketrains.DistLine(is_p, addr, v, ProbEstimates.Spiketrains.CountAssembly())
                ) for v in addr_to_domain[addr]
            ]
        )
        for (is_p, addr) in all_dists_to_show
    ]

    ### Generate the figure.
    (groups_to_show, meta_labels_to_show) = ProbEstimates.Spiketrains.multiparticle_scores_groups(
        keys(doms), values(doms),
        max_weight_idx_at_each_time[1],
        sort(unique(max_weight_idx_at_each_time)),
        dists_vals_to_show,
        neurons_to_show_indices
    )
    return (ProbEstimates.Spiketrains.draw_multiparticle_multistep_spiketrain_group_fig_plus_extras(
        (vcat(full_specs, groups_to_show), meta_labels_to_show, length(full_specs)),
        trs_at_each_time, logweights_at_each_time,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order, addr_to_domain);
        timestep_length_to_latency_ratio=5/3,
        figure_title="Spikes from SMC Neurons for 3D Tracking",
        kwargs...
    ), full_specs)
end

### Get the particle weights and traces at each timestep
weighted_traces = first(weighted_traces_vec)
logweights_at_each_time = [[logweight for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_traces ]
traces_at_each_time = [[trace for (trace, logweight) in weighted_traces_at_time] for weighted_traces_at_time in weighted_traces ]


function augment_figure_with_wave!(f, times, axissize, n_hidden_lines, axispos)
    all_times = sort(collect(Iterators.flatten(times[1:n_hidden_lines])))
    count_in_window(t, w) = count(x -> t - w ≤ x ≤ t, all_times)

    xs = 10:0.05:150
    ax = Axis(f[axispos, 1], title="Overall L4 activity (10ms average)")
    hideydecorations!(ax)
    # hidexdecorations!(ax)
    ax.xticks = (0:25:axissize, ["$x" for x in 0:25:axissize])
    ax.xticksvisible=false
    ax.xticklabelsvisible=false
    lines!(ax, xs, map(x -> count_in_window(x, 10), xs), color=:black, linewidth=6)
    onany(t) do t # update the limits at the given times
        xlims!(ax, (t[], t[] + axissize))
    end
    t[] = 0
    rowsize!(f.layout, 1, Relative(.3))
    ax.tellheight = true
    f
end

### Make the Makie figure with the animation.
### `t` controls which window of time is shown on screen.
AXISSIZE = 150
θs_to_view() = [-0.4, 0.0, 0.4]
(((f, t), (times, group_labels, colors, n_hidden_lines)), hidden_line_specs) = make_Qθ_spiketrain_fig_and_get_all_L4(
    traces_at_each_time[2:end], logweights_at_each_time[2:end],
    θs_to_view(), 1:100;
    figure_title="Spikes from SMC Neurons for 3D Tracking",
    resolution=(750, 150), return_metadata=true,
    first_label_length=170, axissize=AXISSIZE
); t[] = 45; GLMakie.activate!(); f

f = Figure(resolution=(1200, 500))
ax = Axis(f[2, 1], title="Spikes from neurons tuned to θ ∈ $(θs_to_view())")
ax.xlabel = "Prey θ position on retina"
ax.xticks = (0:25:175, ["$(round(x, digits=1))" for x in STARTθ:0.2:(STARTθ+1.41)])
xlims!(ax, (0, 150))
hideydecorations!(ax)
f
# draw_lines!(ax, lines, labels, colors, time, xmin, xmax, axissize; hide_y_decorations=true)

sspikes = times[n_hidden_lines+1:end] # all spikelines to show

spikes_to_show=[sort(vcat(sspikes[(5*(x-1)+1):5*x]...)) for x=1:20] # concatenate every group of 10 spiketrains

spikes_to_show = [
    [sort(vcat(sspikes[(st + 5*(x-1)+1):(st + 5*x)]...)) for x=1:20]
    for st = 0:100:200
]

colors = vcat(([col for _ in lst] for (col, lst) in zip((:blue, :black, :red), spikes_to_show))...)
ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(ax,
    vcat(spikes_to_show...), [], colors,
    t, nothing, nothing, 150
)

c = Observable(RGBA(colorant"yellow", 0.2))
poly!(ax, Rect2D(Point2(75, 0), Point2(16, 61)), color=c)

f

augment_figure_with_wave!(f, times, AXISSIZE, n_hidden_lines, 1)
f

### Below this is for a different plot than above ###

function get_window(spikes, (st, nd))
    [
        filter(x -> st ≤ x ≤ nd, train)
        for train in spikes
    ]
end
function plot_spikefraction_plot(traces_at_each_time, hidden_line_specs, times, first_timestep_plotted)
    times_for_theta_values = times[1:length(hidden_line_specs[1].line_specs)]
    time_to_probs = [
        get_choices(first(traces))[:steps => get_args(first(traces))[1] => :latents => :true_θ => :proposal_probs]
        for traces in traces_at_each_time
    ]

    points = Point2[]
    for (probs, traces) in zip(time_to_probs, traces_at_each_time)
        t = get_args(first(traces))[1]
        plottime = t - first_timestep_plotted
        plottime < 0 && continue
        window = get_window(times_for_theta_values, (25*plottime, 25*(plottime+1)))
        subwindow = get_window(times_for_theta_values, (25*plottime, 25*(plottime+2/5)))
        total_n_spikes = reduce(+, length.(window), init=0.)
        for (i, prob) in enumerate(probs)
            push!(points, Point2(prob, length(subwindow[i])/ total_n_spikes))
        end
    end

    f = Figure()
    ax = Axis(f[1, 1])
    scatter!(ax, points)

    return (f, ax)
end
(f, ax) = plot_spikefraction_plot(traces_at_each_time, hidden_line_specs, times, 2)
f