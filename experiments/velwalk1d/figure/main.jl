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

function make_layout(f, fpos)
    f[fpos...] = GridLayout()
end

make_exact_bayes_filter_heatmaps!(layout, gt_tr) =
    draw_2d_posterior!(layout, get_enumeration_grids(gt_tr), gt_tr; show_statistics=false, show_colorbars=false)

get_pos(tr) = latents_choicemap(tr, get_args(tr)[1])[:xₜ => :val]
get_vel(tr) = latents_choicemap(tr, get_args(tr)[1])[:vₜ => :val]
vel_to_idx(v) = v - first(Vels()) + 1
function draw_particles_visualization!(layout, inferred_trs)
    n_particles = length(first(inferred_trs))
    colors = vcat([:red, :blue], [:black for _=1:(n_particles - 2)])
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

    draw_particle_squares!(layout, pos_particles, vel_particles, n_particles)
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

    pos_lines = [[] for _ in Positions() for _=1:n_particles]
    vel_lines = [[] for _ in Vels() for _=1:n_particles]
    posval_specs = [ProbEstimates.Spiketrains.VarValLine(:xₜ, pos) for pos in Positions()]
    varval_specs = [ProbEstimates.Spiketrains.VarValLine(:vₜ, v) for v in Vels()]
    specs = vcat(posval_specs, varval_specs)

    starttime = 0
    for (t_plus_1, particles) in enumerate(inferred_trs)
        liness = [
            ProbEstimates.Spiketrains.get_lines(specs, tr, spiketrain_data_args; nest_all_at=nestat_addr(t_plus_1 - 1))
            for (tr, _) in particles
        ]
        pos_lines_at_this_time = [liness[particle][val] for val in Positions() for particle in 1:n_particles]
        vel_lines_at_this_time = [liness[particle][length(Positions()) + vel_to_idx(v)] for v in Vels() for particle in 1:n_particles]

        for (pos_line, pos_line_now) in zip(pos_lines, pos_lines_at_this_time)
            append!(pos_line, pos_line_now .+ starttime)
        end
        for (vel_line, vel_line_now) in zip(vel_lines, vel_lines_at_this_time)
            append!(vel_line, vel_line_now .+ starttime)
        end
        starttime += time_per_step
    end

    return (pos_lines, vel_lines)
end

function draw_value_spiketrains!(layout, time_per_step, gt_tr, inferred_trs)
    T = get_args(gt_tr)[1]

    min_time = 0
    max_time = time_per_step * (T + 1)

    velax = Axis(layout[1, 1], xlabel="Time (ms)", ylabel="Neurons for Velocity Value")
    posax = Axis(layout[2, 1],  xlabel="Time (ms)", ylabel="Neurons for Position Value")
    rowsize!(layout, 1, Relative(1/3))

    xlims!(velax, (min_time, max_time))
    xlims!(posax, (min_time, max_time))

    n_particles = length(first(inferred_trs))

    poscolors = vcat(
        (vcat([:red, :blue], [:black for _=1:(n_particles-2)])
        for _ in Positions())...
    )
    velcolors = vcat(
        (vcat([:red, :blue], [:black for _=1:(n_particles-2)])
        for _ in Vels())...
    )

    (pos_lines, vel_lines) = get_pos_vel_value_spiketrains(time_per_step, inferred_trs)

    #                                                  (ax, lines, labels, colors, time, xmin, xmax)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(velax, reverse(vel_lines), [], reverse(velcolors), 0, min_time, max_time; hide_y_decorations = false)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(posax, reverse(pos_lines), [], reverse(poscolors), 0, min_time, max_time; hide_y_decorations = false)

    ylims!(velax, (0, n_particles * length(Vels()) + 1))
    ylims!(posax, (0, n_particles * length(Positions()) + 1))
    # velax.yticks = (
    #     (n_particles/2):n_particles:((length(Vels()) - 1/2) * n_particles),
    #     ["$i" for i in Vels()]
    # )
    # posax.yticks = (
    #     (n_particles/2):n_particles:((length(Positions()) - 1/2) * n_particles),
    #     ["$i" for i in Positions()]
    # )

    for (ax, varvals) in [(velax, Vels()), (posax, Positions())]
        ax.yticks = ((n_particles/2):n_particles:((length(varvals) - 1/2) * n_particles), ["$i" for i in varvals])
        ax.ygridvisible = false
        ax.yminorgridvisible = true
        ax.yminorticks = IntervalsBetween(2)
    end
end

#=
inferred_trs is a vector [[(tr, log_importance_weight) for _=1:n_particles] for _=1:n_timesteps]
The first 2 particles at each timestep are "distinguished".
=#
function make_figure(gt_tr, inferred_trs)
    f = Figure(;resolution=(1000, 1400))
    
    make_exact_bayes_filter_heatmaps!(make_layout(f, (1, 1)), gt_tr)
    draw_particles_visualization!(make_layout(f, (2, 1)), inferred_trs)
    make_layout(f, (3, 1))
    rowsize!(f.layout, 3, Relative(1/3))
    draw_value_spiketrains!(make_layout(f, (3, 1)), 200, gt_tr, inferred_trs)
    # draw_score_spiketrains!(make_layout(f, (4, 1)), inferred_trs)

    f
end
make_figure(gt_tr; n_particles=10) = make_figure(gt_tr, 
    smc(gt_tr, n_particles, exact_init_proposal, exact_step_proposal;
        ess_threshold = -Inf # no resampling
    )[2]
)
make_figure(; n_particles=10, n_steps=6) = make_figure(generate(model, (n_steps,))[1]; n_particles)

make_figure(gt_tr; n_particles=3)