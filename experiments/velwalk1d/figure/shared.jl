using CairoMakie
using DynamicModels
using Colors

set_theme!(font="Arial")

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

function to_obs_nest_addr(p::Pair)
    if p.first == :init
        @assert p.second == :latents
        return :init => :obs
    else
        @assert p.first == :steps 
        (t, rest) = p.second 
        @assert rest == :latents 
        return :steps => t => :obs 
    end
end

nestat_addr(t) =
    if t == 0
        :init => :latents
    else
        :steps => t => :latents
    end

function make_layout(f, fpos; title=nothing)
    layout = f[fpos...] = GridLayout()

    if !isnothing(title)
        Label(layout[1, 1, Top()], title, textsize=26, padding=(0, 0, 20, 0))
    end

    return layout
end

relative_vel_size() = length(Vels()) / (length(Vels()) + length(Positions()))
function setup_vel_pos_axes(layout)
    posax = Axis(layout[1, 1], ylabel="Position")
    velax = Axis(layout[2, 1], ylabel="Velocity", xlabel="Time")
    linkxaxes!(velax, posax)
    rowsize!(layout, 2, Relative(relative_vel_size()))
    posax.xticksvisible = false
    posax.xticklabelsvisible = false
    velax.xlabelpadding = -5.0
    rowgap!(layout, 10)
    ylims!(velax, (first(Vels()) - 0.5, last(Vels()) + 0.5))
    ylims!(posax, (first(Positions()) - 0.5, last(Positions()) + 0.5))
    posax.xgridvisible = false
    posax.ygridvisible = false
    velax.xgridvisible = false
    velax.ygridvisible = false
    return (velax, posax)
end

relative_vel_size_withobs() = length(Vels()) / (length(Vels()) + 2*length(Positions()))
relative_posobs_size() = length(Positions()) / (length(Vels()) + 2*length(Positions()))
function setup_obs_pos_vel_axes(layout)
    obsax = Axis(layout[1, 1], ylabel="yᵈₜ Neurons")
    posax = Axis(layout[2, 1], ylabel="xₜ Neurons")
    velax = Axis(layout[3, 1], ylabel="vₜ Neurons")

    linkxaxes!(obsax, posax, velax)

    rowsize!(layout, 1, Relative(relative_posobs_size()))
    rowsize!(layout, 2, Relative(relative_posobs_size()))
    rowsize!(layout, 3, Relative(relative_vel_size_withobs()))

    rowgap!(layout, 10)

    obsax.xticksvisible = false
    obsax.xticklabelsvisible = false
    posax.xticksvisible = false
    posax.xticklabelsvisible = false
    velax.xlabelpadding = -5.0
    
    ylims!(velax, (first(Vels()) - 0.5, last(Vels()) + 0.5))
    ylims!(posax, (first(Positions()) - 0.5, last(Positions()) + 0.5))
    ylims!(obsax, (first(Positions()) - 0.5, last(Positions()) + 0.5))
    
    obsax.xgridvisible = false
    obsax.ygridvisible = false
    posax.xgridvisible = false
    posax.ygridvisible = false
    velax.xgridvisible = false
    velax.ygridvisible = false
    
    return (obsax, posax, velax)
end

get_pos(tr) = latents_choicemap(tr, get_args(tr)[1])[:xₜ => :val]
get_vel(tr) = latents_choicemap(tr, get_args(tr)[1])[:vₜ => :val]
get_obsd(tr) = latents_choicemap(tr, get_args(tr)[1])[:yᵈₜ => :val]
vel_to_idx(v) = v - first(Vels()) + 1

function get_particle_weights_colors(inferred_trs)
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

    return (pos_particles, vel_particles)
end

function get_particle_weights_colors_withobs(inferred_trs)
    n_particles = length(first(inferred_trs))
    colors = vcat([:blue, :red], [:black for _=1:(n_particles - 2)])
    n_particles = length(first(inferred_trs))
    pos_particles = []
    vel_particles = []
    obs_particles = []

    for trs_and_weights in inferred_trs
        @assert length(trs_and_weights) == n_particles

        poss = [[] for _ in Positions()]
        vels = [[] for _ in Vels()]
        obss = [[] for _ in Positions()]
        push!(pos_particles, poss)
        push!(vel_particles, vels)
        push!(obs_particles, obss)

        trs = map(x -> x[1], trs_and_weights)
        logweights = map(x -> x[2], trs_and_weights)
        normalized_weights = exp.(logweights .- logsumexp(logweights))
        positions = map(get_pos, trs)
        velocities = map(get_vel, trs)
        observations = map(get_obsd, trs)
        
        for (wt, color, pos, vel, o) in zip(normalized_weights, colors, positions, velocities, observations)
            push!(poss[pos], (wt, color))
            push!(vels[vel_to_idx(vel)], (wt, color))
            push!(obss[o], (wt, color))
        end
    end

    return (obs_particles, pos_particles, vel_particles)
end

function get_pos_vel_value_spiketrains(time_per_step, inferred_trs;
    # used to control how many auto-normalization spikes are produced:
    num_autonormalization_spikes_for_each_timestep = [nothing for _ in inferred_trs]
)
    n_particles = length(first(inferred_trs))
    propose_sampling_tree = Dict(:vₜ => [], :xₜ => [:vₜ])
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
        # println("t = $t")

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
            other_factors_to_multiply_in=exp.(previous_normalized_logweights),
            num_autonormalization_spikes=num_autonormalization_spikes_for_each_timestep[t_plus_1]
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

function get_obs_pos_vel_value_spiketrains(time_per_step, inferred_trs;
    # used to control how many auto-normalization spikes are produced:
    num_autonormalization_spikes_for_each_timestep = [nothing for _ in inferred_trs]
)
    T = get_args(gt_tr)[1] # we will get the spiketrains for the last timestep in the trace
    n_particles = length(first(inferred_trs))
    propose_sampling_tree = Dict(:yᵈₜ => [],  :vₜ => [:yᵈₜ], :xₜ => [:vₜ])
    assess_sampling_tree = Dict(:vₜ => [], :xₜ => [:vₜ], :yᵈₜ => [:xₜ])
    propose_topological_order = [:yᵈₜ, :vₜ, :xₜ]
    spiketrain_data_args = (propose_sampling_tree, assess_sampling_tree, propose_topological_order)

    ### line specs
    obsval_specs = [ProbEstimates.Spiketrains.VarValLine(:yᵈₜ, pos) for pos in Positions()]
    posval_specs = [ProbEstimates.Spiketrains.VarValLine(:xₜ, pos) for pos in Positions()]
    varval_specs = [ProbEstimates.Spiketrains.VarValLine(:vₜ, v) for v in Vels()]

    normalized_weight_specs = [ProbEstimates.Spiketrains.NormalizedWeight(i) for i=1:n_particles]
    normalization_line_spec = ProbEstimates.Spiketrains.LogNormalization()

    obs_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for linespec in obsval_specs for particle_idx=1:n_particles
    ]
    pos_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for linespec in posval_specs for particle_idx=1:n_particles
    ]
    vel_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for linespec in varval_specs for particle_idx=1:n_particles
    ]

    all_specs = vcat(obs_particle_linespecs, pos_particle_linespecs, vel_particle_linespecs, normalized_weight_specs, [normalization_line_spec])

    # display(all_specs)
    # display(vcat(
    #     [[(;var="obs", val=v, i=i)] for i=1:n_particles for v in Positions()],
    #     [[(;var="pos", val=v, i=i)] for i=1:n_particles for v in Positions()],
    #     [[(;var="vel", val=v, i=i)]  for i=1:n_particles for v in Vels()],
    #     [[(;var="normwt", i=i)] for i=1:n_particles],
    #     [(;var="-logz")]
    # ))

    ### get lines for each timestep of inference
    obs_lines = [[] for _ in Positions() for _=1:n_particles ]
    pos_lines = [[] for _ in Positions() for _=1:n_particles ]
    vel_lines = [[] for _ in Vels() for _=1:n_particles ]
    normalized_weight_lines = [[] for _=1:n_particles]
    normalization_line = []
    starttime = 0
    for (t_plus_1, particles) in enumerate(inferred_trs)
        t = t_plus_1 - 1
        # println("t = $t")

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
            other_factors_to_multiply_in=exp.(previous_normalized_logweights),
            num_autonormalization_spikes=num_autonormalization_spikes_for_each_timestep[t_plus_1],
            vars_disc_to_cont=Dict(:yᵈₜ => (nest_addr -> (ProbEstimates.Spiketrains.nest(to_obs_nest_addr(nest_addr), :yᶜₜ))))
        )

        for (line, line_now) in zip(
            vcat(obs_lines, pos_lines, vel_lines, normalized_weight_lines, [normalization_line]),
            lines_at_this_time
        )
            append!(line, line_now .+ starttime)
        end

        starttime += time_per_step
    end

    return (obs_lines, pos_lines, vel_lines, normalized_weight_lines, normalization_line)
end

function draw_weight_spiketrains_2!(layout, (min_time, max_time), normalized_weight_lines, normalization_line)
    weightax = Axis(layout[1, 1], ylabel="Particle Index")
    logax = Axis(layout[2, 1], xlabel="Time (ms)")
    linkxaxes!(weightax, logax)
    rowsize!(layout, 1, Relative(0.85))
    weightax.xticksvisible = false
    weightax.xticklabelsvisible = false
    logax.xlabelpadding = -5.0
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

    return (weightax, logax)
end

function draw_obs_particles!(layout, tr, inferred_trs)
    axislayout = layout[1, 1] = GridLayout()
    (velax, posax) = setup_vel_pos_axes(axislayout)
    n_particles = length(first(inferred_trs))

    xlims!(posax, (0., 3.))
    xlims!(velax, (0., 3.))

    # draw inferred trs
    (pos_particles, vel_particles) = get_particle_weights_colors(inferred_trs)
    particles = draw_particle_squares_for_variable!(posax, Positions(), pos_particles, n_particles)
    # println(particles)
    draw_particle_squares_for_variable!(velax, Vels(), vel_particles, n_particles)    

    # draw obs
    times = 0:(get_args(tr)[1])
    pos_observations = [
        (
            let cm = obs_choicemap(tr, t)
                let (key, _) = first(get_submaps_shallow(cm))
                    obs_choicemap(tr, t)[key => :val]
                end
            end
        )
        for t in times
    ]
    obs = lines!(posax, times, pos_observations, color=:seagreen, linewidth=5)#, markersize=20, marker='■')

    l = Legend(layout[2, 1], [obs, particles], ["Observed Positions", "Inferred Particles"], orientation=:horizontal)
    l.tellwidth=false
    l.tellheight=true
    rowgap!(layout, 10)
end