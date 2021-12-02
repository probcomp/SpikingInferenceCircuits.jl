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
include("shared.jl")

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
    (pos_particles, vel_particles) = get_particle_weights_colors(inferred_trs)
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
function make_figure_2(gt_tr, inferred_trs; time_per_step=200, start_caption_letter = 'e')
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

    subcaption_padding = (0, 0, 5, 45)
    subcaption_align = :center

    letters = [start_caption_letter + i for i=0:4]
    Label(f.layout[1, 1, Bottom()], "($(letters[1])) Trajectory & Observations", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[2, 1, Bottom()], "($(letters[2])) Exact Filtering Posterior", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[3, 1, Bottom()], "($(letters[3])) Inferred Particles", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[4, 1, Bottom()], "($(letters[4])) Value Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[5, 1, Bottom()], "($(letters[5])) Weight Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)

    draw_traj_obs!(traj_obs_layout, gt_tr)
    ax = draw_filtering_posterior!(posterior_layout, gt_tr)
    draw_inferred_particles!(particle_layout, inferred_trs)
    draw_value_weight_spiketrains!(value_spiketrain_layout, weight_spiketrain_layout, gt_tr, [trs for trs in inferred_trs])

    (f, ax)
end
gt_tr = generate(model, (10,), choicemap(
    (:init => :latents => :xₜ => :val, 1),
    (:init => :latents => :vₜ => :val, 1)
))[1];
inferred_trs = smc(gt_tr, 10, exact_init_proposal, approx_step_proposal;
    ess_threshold = -Inf # no resampling
)[2];
(f, ax) = make_figure_2(gt_tr, inferred_trs); f