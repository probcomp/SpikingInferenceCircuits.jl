
make_exact_bayes_filter_heatmaps!(layout, gt_tr) =
draw_2d_posterior!(layout, get_enumeration_grids(gt_tr), gt_tr; show_statistics=false, show_colorbars=false)

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

    lines_at_this_time = ProbEstimates.Spiketrains.get_lines_for_multiparticle_specs(
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
