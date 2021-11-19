#=
--- In illustrator ---
Pseudocode for model & inference, and Graphical Model
----------------------
--- Julia code produces ---
Top:  Bayes Filter
Next: Particle representation of inference results, with 2 distinguished (colored) particles
Next: Spiketrains showing position & velocity value for each particle, with distinguished spiketrains colored
Next: Spiketrains for post-multiplication score-line for each particle, with colored distinguished particles.
----------------------
=#
using DynamicModels

include("../model.jl")
Positions() = 1:10
SwitchProb() = 0.0
# include("pm_model.jl")
include("../inference.jl")
include("../visualize.jl")
ProbEstimates.DoRecipPECheck() = false
include("../utils.jl")
ProbEstimates.use_noisy_weights!()
ProbEstimates.AutonormalizeRepeaterRate() = 0.25
ProbEstimates.AutonormalizeSpeedupFactor() = 1.5

function make_layout(f, fpos; title=nothing)
    layout = f[fpos...] = GridLayout()

    if !isnothing(title)
        Label(layout[1, 1, Top()], title, textsize=26, padding=(0, 0, 20, 0))
    end

    return layout
end

make_exact_bayes_filter_heatmaps!(layout, gt_tr) =
    draw_2d_posterior!(layout, get_enumeration_grids(gt_tr), gt_tr; show_statistics=false, show_colorbars=false)

get_pos(tr) = latents_choicemap(tr, get_args(tr)[1])[:xₜ => :val]
get_vel(tr) = latents_choicemap(tr, get_args(tr)[1])[:vₜ => :val]
vel_to_idx(v) = v - first(Vels()) + 1
function draw_particles_visualization!(layout, inferred_trs)
    n_particles = length(first(inferred_trs))
    colors = vcat([:blue, :red], [:black for _=1:(n_particles - 2)])
    pos_particles = []
    vel_particles = []

    for trs_and_weights in inferred_trs
        @assert length(trs_and_weights) == n_particles

        poss = [[] for _ in Positions()]
        vels = [[] for _ in Vels()]
        push!(pos_particles, poss)
        push!(vel_particles, vels)

        trs = map(x -> x[1], trs_and_weights)
        logweights = map(x -> x[2], trs_and_weights)
        normalized_weights = exp.(logweights .- logsumexp(logweights))
        positions = map(get_pos, trs)
        velocities = map(get_vel, trs)
        
        for (wt, color, pos, vel) in zip(normalized_weights, colors, positions, velocities)
            push!(poss[pos], (wt, color))
            push!(vels[vel_to_idx(vel)], (wt, color))
        end
    end

    draw_particles!(layout, pos_particles, vel_particles, n_particles)
end

nestat_addr(t) =
    if t == 0
        :init => :latents
    else
        :steps => t => :latents
    end
function get_pos_vel_value_spiketrains(time_per_step, inferred_trs)
    n_particles = length(first(inferred_trs))
    propose_sampling_tree = Dict(:xₜ => [], :vₜ => [:xₜ])
    assess_sampling_tree = Dict(:vₜ => [], :xₜ => [:vₜ])
    propose_topological_order = [:xₜ, :vₜ]
    spiketrain_data_args = (propose_sampling_tree, assess_sampling_tree, propose_topological_order)

    ### line specs
    posval_specs = [ProbEstimates.Spiketrains.VarValLine(:xₜ, pos) for pos in Positions()]
    varval_specs = [ProbEstimates.Spiketrains.VarValLine(:vₜ, v) for v in Vels()]

    normalized_weight_specs = [ProbEstimates.Spiketrains.NormalizedWeight(i) for i=1:n_particles]
    normalization_line_spec = ProbEstimates.Spiketrains.LogNormalization()

    pos_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for linespec in posval_specs for particle_idx=1:n_particles
    ]
    vel_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for linespec in varval_specs for particle_idx=1:n_particles
    ]

    all_specs = vcat(pos_particle_linespecs, vel_particle_linespecs, normalized_weight_specs, [normalization_line_spec])

    ### get lines for each timestep of inference
    pos_lines = [[] for _ in Positions() for _=1:n_particles]
    vel_lines = [[] for _ in Vels() for _=1:n_particles]
    normalized_weight_lines = [[] for _=1:n_particles]
    normalization_line = []
    starttime = 0
    for (t_plus_1, particles) in enumerate(inferred_trs)
        t = t_plus_1 - 1
        println("t = $t")

        traces = map(x -> x[1], particles)

        previous_logweights = 
            if t == 0
                [0. for _=1:n_particles]
            else
                map(x -> x[2], inferred_trs[t])
            end
        current_logweights = map(x -> x[2], particles)

        previous_normalized_logweights = previous_logweights .- logsumexp(previous_logweights)
        log_weight_updates = current_logweights .- previous_logweights

        lines_at_this_time = ProbEstimates.Spiketrains.get_lines_for_particles(
            all_specs,
            traces, # traces
            log_weight_updates, # log_weight_updates which should be computed
            spiketrain_data_args;
            nest_all_at=nestat_addr(t),
            other_factors_to_multiply_in=exp.(previous_normalized_logweights)
        )

        for (line, line_now) in zip(
            vcat(pos_lines, vel_lines, normalized_weight_lines, [normalization_line]),
            lines_at_this_time
        )
            append!(line, line_now .+ starttime)
        end

        starttime += time_per_step
    end

    return (pos_lines, vel_lines, normalized_weight_lines, normalization_line)
end

function draw_value_spiketrains!(layout, (min_time, max_time), pos_spiketrains, vel_spiketrains)
    velax = Axis(layout[1, 1], xlabel="Time (ms)", ylabel="Velocity Neurons")
    posax = Axis(layout[2, 1],  xlabel="Time (ms)", ylabel="Position Neurons")
    rowsize!(layout, 1, Relative(1/3))

    xlims!(velax, (min_time, max_time))
    xlims!(posax, (min_time, max_time))

    n_particles = length(first(inferred_trs))

    poscolors = vcat(
        (vcat([:blue, :red], [:black for _=1:(n_particles-2)])
        for _ in Positions())...
    )
    velcolors = vcat(
        (vcat([:blue, :red], [:black for _=1:(n_particles-2)])
        for _ in Vels())...
    )

    #                                                  (ax, lines, labels, colors, time, xmin, xmax)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(velax, reverse(vel_spiketrains), [], reverse(velcolors), 0, min_time, max_time; hide_y_decorations = false)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(posax, reverse(pos_spiketrains), [], reverse(poscolors), 0, min_time, max_time; hide_y_decorations = false)

    ylims!(velax, (0, n_particles * length(Vels()) + 1))
    ylims!(posax, (0, n_particles * length(Positions()) + 1))

    for (ax, varvals) in [(velax, Vels()), (posax, Positions())]
        ax.yticks = ((n_particles/2):n_particles:((length(varvals) - 1/2) * n_particles), ["$i" for i in varvals])
        ax.ygridvisible = false
        ax.yminorgridvisible = true
        ax.yminorticks = IntervalsBetween(2)
    end
end

function draw_score_spiketrains!(layout, (min_time, max_time), normalized_weight_spiketrains, normalization_line_train)
    n_particles = length(normalized_weight_spiketrains)
    weight_ax = layout[1, 1] = Axis(layout[1, 1])
    normalized_weight_colors = vcat([:blue, :red], [:black for _=1:(n_particles-2)])
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(weight_ax, reverse(normalized_weight_spiketrains), [], reverse(normalized_weight_colors), 0, min_time, max_time; hide_y_decorations = false)
    xlims!(weight_ax, (min_time, max_time))

    norm_ax = layout[2, 1] = Axis(layout[2, 1])
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(norm_ax, [normalization_line_train], [], [:black], 0, min_time, max_time; hide_y_decorations = false)
    xlims!(norm_ax, (min_time, max_time))
end

#=
inferred_trs is a vector [[(tr, log_importance_weight) for _=1:n_particles] for _=1:n_timesteps]
The first 2 particles at each timestep are "distinguished".
=#
function make_figure(gt_tr, inferred_trs; time_per_step=200)
    f = Figure(;resolution=(1000, 1650))
    println("f constructed")

    make_exact_bayes_filter_heatmaps!(make_layout(f, (1, 1); title="Posterior"), gt_tr)
    draw_particles_visualization!(make_layout(f, (2, 1); title="Particles from Inference"), inferred_trs)
    
    println("first 2 visualizations done")

    T = get_args(gt_tr)[1]
    ms_timerange = (0, time_per_step * (T + 1))
    (pos_lines, vel_lines, normalized_weight_lines, normalization_line) = get_pos_vel_value_spiketrains(time_per_step, inferred_trs);

    display(normalized_weight_lines)

    println("got spiketrains")

    draw_value_spiketrains!(
        make_layout(f, (3, 1); title="Spiketrains from Particle-Value Assemblies"),
        ms_timerange, pos_lines, vel_lines
    )
    rowsize!(f.layout, 3, Relative(1/3))
    draw_score_spiketrains!(
        make_layout(f, (4, 1); title="Spiketrains from Particle-Weight Assemblies"),
        ms_timerange, normalized_weight_lines, normalization_line
    )

    println("done drawing; outputting figure...")

    f
end
make_figure(gt_tr; n_particles=10) = make_figure(gt_tr, 
    smc(gt_tr, n_particles, exact_init_proposal, exact_step_proposal;
        ess_threshold = -Inf # no resampling
    )[2]
)
make_figure(; n_particles=10, n_steps=6) = make_figure(generate(model, (n_steps,))[1]; n_particles)

# make_figure(gt_tr, inferred_trs)

### DRAFT 2 ###

relative_vel_size() = length(Vels()) / (length(Vels()) + length(Positions()))
function setup_vel_pos_axes(layout)
    velax = Axis(layout[1, 1], ylabel="Velocity")
    posax = Axis(layout[2, 1], ylabel="Position", xlabel="Time")
    linkxaxes!(velax, posax)
    rowsize!(layout, 1, Relative(relative_vel_size()))
    velax.xticksvisible = false
    velax.xticklabelsvisible = false
    posax.xlabelpadding = -2.0
    rowgap!(layout, 10)
    ylims!(velax, (first(Vels()) - 0.5, last(Vels()) + 0.5))
    ylims!(posax, (first(Positions()) - 0.5, last(Positions()) + 0.5))
    posax.xgridvisible = false
    posax.ygridvisible = false
    velax.xgridvisible = false
    velax.ygridvisible = false
    return (velax, posax)
end
function draw_traj_obs!(layout, tr)
    times = 0:(get_args(tr)[1])
    pos_observations = [obs_choicemap(tr, t)[:obs => :val] for t in times]
    gt_pos           = [latents_choicemap(tr, t)[:xₜ => :val] for t in times]
    gt_vel           = [latents_choicemap(tr, t)[:vₜ => :val] for t in times]

    (velax, posax) = setup_vel_pos_axes(layout)

    lines!(velax, times, gt_vel, color=:black)
    lines!(posax, times, gt_pos, color=:black)
    scatter!(posax, times, pos_observations, color=:seagreen, markersize=12)
end
function draw_filtering_posterior!(layout, tr)
    times = 0:(get_args(tr)[1])
    pos_observations = [obs_choicemap(tr, t)[:obs => :val] for t in times]
    
    filter_results = get_enumeration_grids(gt_tr)
    vel_posterior = [sum(grid, dims=1) |> normalize |> to_vect for grid in filter_results]
    pos_posterior = [sum(grid, dims=2) |> normalize |> to_vect for grid in filter_results]

    vel_inf_matrix = hcat(vel_posterior...) |> transpose
    vel_maxprob = maximum(vel_inf_matrix)

    pos_inf_matrix = hcat(pos_posterior...) |> transpose
    pos_maxprob = maximum(pos_inf_matrix)
    maxprob = max(vel_maxprob, pos_maxprob)

    (velax, posax) = setup_vel_pos_axes(layout)

    vel_hm = heatmap!(velax,
        times, Vels(), vel_inf_matrix,
        colormap=cgrad([:white, :black], [0., 0.4, 1.0]),
        colorrange = (0, min(vel_maxprob + 0.1, 1.0))
    )
    pos_hm = heatmap!(posax,
        times, Positions(), pos_inf_matrix,
        colormap=cgrad([:white, :black], [0., 0.4, 1.0]),
        colorrange = (0, min(pos_maxprob + 0.1, 1.0))
    )
    pos_obs = scatter!(posax, times, pos_observations, color=:seagreen, markersize=12)

    return velax
end

function draw_inferred_particles!(layout, inferred_trs)
    n_particles = length(first(inferred_trs))
    colors = vcat([:blue, :red], [:black for _=1:(n_particles - 2)])
    n_particles = length(first(inferred_trs))
    pos_particles = []
    vel_particles = []

    for trs_and_weights in inferred_trs
        @assert length(trs_and_weights) == n_particles

        poss = [[] for _ in Positions()]
        vels = [[] for _ in Vels()]
        push!(pos_particles, poss)
        push!(vel_particles, vels)

        trs = map(x -> x[1], trs_and_weights)
        logweights = map(x -> x[2], trs_and_weights)
        normalized_weights = exp.(logweights .- logsumexp(logweights))
        positions = map(get_pos, trs)
        velocities = map(get_vel, trs)
        
        for (wt, color, pos, vel) in zip(normalized_weights, colors, positions, velocities)
            push!(poss[pos], (wt, color))
            push!(vels[vel_to_idx(vel)], (wt, color))
        end
    end

    (velax, posax) = setup_vel_pos_axes(layout)
    draw_particle_squares_for_variable!(posax, Positions(), pos_particles, n_particles)
    draw_particle_squares_for_variable!(velax, Vels(), vel_particles, n_particles)
end
function draw_value_spiketrains_2!(layout, (min_time, max_time), pos_spiketrains, vel_spiketrains, n_particles)
    (velax, posax) = setup_vel_pos_axes(layout)
    posax.xlabel = "Time (ms)"
    velax.ylabel = "Velocity Neurons"
    posax.ylabel = "Position Neurons"

    xlims!(velax, (min_time, max_time))
    xlims!(posax, (min_time, max_time))

    poscolors = vcat(
        (vcat([:blue, :red], [:black for _=1:(n_particles-2)])
        for _ in Positions())...
    )
    velcolors = vcat(
        (vcat([:blue, :red], [:black for _=1:(n_particles-2)])
        for _ in Vels())...
    )

    #                                                  (ax, lines, labels, colors, time, xmin, xmax)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(velax, reverse(vel_spiketrains), [], reverse(velcolors), 0, min_time, max_time; hide_y_decorations = false)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(posax, reverse(pos_spiketrains), [], reverse(poscolors), 0, min_time, max_time; hide_y_decorations = false)

    ylims!(velax, (0, n_particles * length(Vels()) + 1))
    ylims!(posax, (0, n_particles * length(Positions()) + 1))

    for (ax, varvals) in [(velax, Vels()), (posax, Positions())]
        ax.yticks = ((n_particles/2):n_particles:((length(varvals) - 1/2) * n_particles), ["$i" for i in varvals])
        ax.ygridvisible = false
        ax.yminorgridvisible = true
        ax.yminorticks = IntervalsBetween(2)
    end
end
function draw_weight_spiketrains_2!(layout, (min_time, max_time), normalized_weight_lines, normalization_line)
    weightax = Axis(layout[1, 1], ylabel="Particle Index")
    logax = Axis(layout[2, 1], xlabel="Time (ms)")
    linkxaxes!(weightax, logax)
    rowsize!(layout, 1, Relative(0.85))
    weightax.xticksvisible = false
    weightax.xticklabelsvisible = false
    logax.xlabelpadding = -2.0
    rowgap!(layout, 10)
    ylims!(weightax, (0, length(normalized_weight_lines) + 1))
    ylims!(logax, (0, 1))
    weightax.xgridvisible = false
    weightax.ygridvisible = false
    logax.xgridvisible = false
    logax.ygridvisible = false
    logax.yticksvisible = false
    logax.yticklabelsvisible = false

    n_particles = length(normalized_weight_lines)
    normalized_weight_colors = vcat([:blue, :red], [:black for _=1:(n_particles-2)])
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(weightax, reverse(normalized_weight_lines), [], reverse(normalized_weight_colors), 0, min_time, max_time; hide_y_decorations = false)
    xlims!(weightax, (min_time, max_time))

    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(logax, [normalization_line], [], [:black], 0, min_time, max_time; hide_y_decorations = false)
    xlims!(logax, (min_time, max_time))
end
function draw_value_weight_spiketrains!(value_layout, weight_layout, gt_tr, inferred_trs; time_per_step=200)
    n_particles = length(first(inferred_trs))
    T = get_args(gt_tr)[1]
    ms_timerange = (0, time_per_step * (T + 1))
    (pos_lines, vel_lines, normalized_weight_lines, normalization_line) = get_pos_vel_value_spiketrains(time_per_step, inferred_trs);

    println("got spiketrains")
    draw_value_spiketrains_2!(value_layout, ms_timerange, pos_lines, vel_lines, n_particles)

    draw_weight_spiketrains_2!(weight_layout, ms_timerange, normalized_weight_lines, normalization_line)
end

"""
Reorder the particles so that the 2 traces which have the highest
weights later into inference are put first.
"""
function move_highweight_trs_first(inferred_trs)
    laststep_trs = last(inferred_trs)
    n_particles = length(laststep_trs)
    logweights = [wt for (_, wt) in laststep_trs]
    sortedperm = sortperm(logweights)
    maxidx = sortedperm[end]
    nextidx = sortedperm[end-1]
    
    outputperm = [
        maxidx,
        nextidx,
        (
            i for i in 1:n_particles
            if !(i in (maxidx, nextidx))
        )...
    ]
    return [
        trs[outputperm] for trs in inferred_trs
    ]
end
function make_figure_2(gt_tr, inferred_trs; time_per_step=200)
    inferred_trs = move_highweight_trs_first(inferred_trs)

    # Page = 8.5 x 11 in
    # margins: .75 on each side
    # remaining space: 7 x 9.5
    # 1/3 the space for the LHS of the figure
    # so we have 7 * 2/3
    f = Figure(;resolution=(7 * 2/3 * 180, 9.5 * 180))

    traj_obs_layout = f[1, 1] = GridLayout()
    posterior_layout = f[2, 1] = GridLayout()
    particle_layout = f[3, 1] = GridLayout()
    value_spiketrain_layout = f[4, 1] = GridLayout()
    weight_spiketrain_layout = f[5, 1] = GridLayout()

    # Set relative heights of different parts of the figure
    relative_heights = [1, 1, 1, 1.75, 0.75]
    normalized_heights = relative_heights / sum(relative_heights)
    for (i, h) in enumerate(normalized_heights)
        rowsize!(f.layout, i, Relative(h))
    end

    subcaption_padding = (-25, 0, 5, 30)
    subcaption_align = :left
    Label(f.layout[1, 1, Bottom()], "(c) Trajectory & Observations", textsize=20, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[2, 1, Bottom()], "(d) Exact Filtering Posterior", textsize=20, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[3, 1, Bottom()], "(e) Inferred Particles", textsize=20, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[4, 1, Bottom()], "(f) Value Spiketrains", textsize=20, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[5, 1, Bottom()], "(g) Weight Spiketrains", textsize=20, padding=subcaption_padding, halign=subcaption_align)

    draw_traj_obs!(traj_obs_layout, gt_tr)
    ax = draw_filtering_posterior!(posterior_layout, gt_tr)
    draw_inferred_particles!(particle_layout, inferred_trs)
    draw_value_weight_spiketrains!(value_spiketrain_layout, weight_spiketrain_layout, gt_tr, [trs for trs in inferred_trs])

    (f, ax)
end
(f, ax) = make_figure_2(gt_tr, inferred_trs); f