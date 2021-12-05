#=
TODOs - 
- Draft of figure without correct continuous-noise model [if this is possible without things crashing]
- Add correct continuous-noise model
=#

# include("shared.jl")
# include("../model_continuous_obs.jl")
# include("../proposals_continuous_obs.jl")

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

function draw_particle_value_spiketrains!(layout, (obs_lines, pos_lines, vel_lines, _, _, _), particleidx; time_per_step=100)
    (obsax, posax, velax) = setup_obs_pos_vel_axes(layout)

    velax.xlabel = "Time (ms)"

    (min_time, max_time) = (0, time_per_step)
    xlims!(obsax, (min_time, max_time))
    xlims!(posax, (min_time, max_time))
    xlims!(velax, (min_time, max_time))

    color = (particleidx == 1) ? :red : (particleidx == 2) ? :blue : :black
    poscolors = [color for _ in Positions()]
    velcolors = [color for _ in Vels()]

    println("particle index = $particleidx; OBS LINES:")
    display(obs_lines)

    #                                                  (ax, lines, labels, colors, time, xmin, xmax)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(obsax, reverse(obs_lines[particleidx]), [], reverse(poscolors), 0, min_time, max_time; hide_y_decorations = false)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(posax, reverse(pos_lines[particleidx]), [], reverse(poscolors), 0, min_time, max_time; hide_y_decorations = false)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(velax, reverse(vel_lines[particleidx]), [], reverse(velcolors), 0, min_time, max_time; hide_y_decorations = false)
    
    for (ax, varvals) in [(velax, Vels()), (posax, Positions()), (obsax, Positions())]
        ylims!(ax, (0, length(varvals) + 1))
        ax.yticks = (1:length(varvals), ["$i" for i in varvals])
        ax.ygridvisible = false
        ax.yminorgridvisible = true
        ax.yminorticks = IntervalsBetween(2)
    end
end

function draw_weight_output_spiketrains!(layout, (_, _, _, normalized_weight_lines, normalization_line, _))
    Axis(layout[1, 1])
    Axis(layout[2, 1])
end
function draw_weight_internals_spiketrains!(layout, (_, _, _, _, _, weight_internal_spiketrains), _)
    Axis(layout[1, 1])
end

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
function get_spiketrains_for_one_timestep_figure(
    gt_tr, inferred_trs;
    num_autonormalization_spikes=nothing,
    n_particle_value_trains
)
    T = get_args(gt_tr)[1] # we will get the spiketrains for the last timestep in the trace
    n_particles = length(first(inferred_trs))
    propose_sampling_tree = Dict(:yᵈₜ => [], :xₜ => [:yᵈₜ], :vₜ => [:xₜ])
    assess_sampling_tree = Dict(:vₜ => [], :xₜ => [:vₜ], :yᵈₜ => [:xₜ])
    propose_topological_order = [:yᵈₜ, :xₜ, :vₜ]
    spiketrain_data_args = (propose_sampling_tree, assess_sampling_tree, propose_topological_order)

    obsval_specs = [ProbEstimates.Spiketrains.VarValLine(:yᵈₜ, pos) for pos in Positions()]
    posval_specs = [ProbEstimates.Spiketrains.VarValLine(:xₜ, pos) for pos in Positions()]
    varval_specs = [ProbEstimates.Spiketrains.VarValLine(:vₜ, v) for v in Vels()]

    normalized_weight_specs = [ProbEstimates.Spiketrains.NormalizedWeight(i) for i=1:n_particles]
    normalization_line_spec = ProbEstimates.Spiketrains.LogNormalization()

    obs_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for particle_idx=1:n_particle_value_trains for linespec in obsval_specs
    ]
    pos_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for particle_idx=1:n_particle_value_trains for linespec in posval_specs
    ]
    vel_particle_linespecs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(particle_idx, linespec)
        for particle_idx=1:n_particle_value_trains for linespec in varval_specs
    ]

    all_specs = vcat(
        obs_particle_linespecs,
        pos_particle_linespecs,
        vel_particle_linespecs,
        normalized_weight_specs,
        [normalization_line_spec]
    )

    # println()
    # println("---SPECS---")
    # display(all_specs[1:20])
    # display(all_specs[21:40])
    # display(all_specs[41:end])
    # println()
    # println()

    obs_lines = [[[ (;var="obs", particle=i, val=pos)  ] for pos in Positions()] for i=1:n_particle_value_trains]
    pos_lines = [[[ (;var="pos", particle=i, val=pos) ] for pos in Positions()] for i=1:n_particle_value_trains]
    vel_lines = [[[ (;var="vel", particle=i, val=vel) ] for vel in Vels()] for i=1:n_particle_value_trains]
    normalized_weight_lines = [[(var=:normalized_weight, particle=i)] for i=1:n_particles]
    normalization_line = [(;var="normalization_line")]
    collectors = vcat(obs_lines..., pos_lines..., vel_lines..., normalized_weight_lines, [normalization_line])

    # println()
    # println("---OUTPUTS---")
    # display(collectors[1:20])
    # display(collectors[21:40])
    # display(collectors[41:end])
    # println()
    # println()

    previous_logweights = map(x -> x[2], inferred_trs[T])
    previous_normalized_logweights = previous_logweights .- logsumexp(previous_logweights)
    current_logweights = map(x -> x[2], inferred_trs[T + 1])
    log_weight_updates = current_logweights .- previous_logweights
    lines = ProbEstimates.Spiketrains.get_lines_for_particles(
        all_specs,
        [tr for (tr, wt) in last(inferred_trs)], # traces
        log_weight_updates, # log_weight_updates which should be computed
        spiketrain_data_args;
        nest_all_at=nestat_addr(T),
        other_factors_to_multiply_in=exp.(previous_normalized_logweights),
        num_autonormalization_spikes,
        vars_disc_to_cont=Dict(:yᵈₜ => (nest_addr -> (ProbEstimates.Spiketrains.nest(to_obs_nest_addr(nest_addr), :yᶜₜ))))
    )

    # println()
    # println()
    # println("---LINES---")
    # display(lines[1:20])
    # display(lines[21:40])
    # display(lines[41:end])
    # println()

    # lines = ProbEstimates.Spiketrains.get_lines_for_particles(
    #     [ProbEstimates.Spiketrains.ParticleLineSpec(1, linespec) for linespec in obsval_specs],
    #     [tr for (tr, wt) in last(inferred_trs)], # traces
    #     log_weight_updates, # log_weight_updates which should be computed
    #     spiketrain_data_args;
    #     nest_all_at=nestat_addr(T),
    #     other_factors_to_multiply_in=exp.(previous_normalized_logweights),
    #     num_autonormalization_spikes,
    #     vars_disc_to_cont=Dict(:yᵈₜ => (nest_addr -> (ProbEstimates.Spiketrains.nest(to_obs_nest_addr(nest_addr), :yᶜₜ))))
    # )

    # display(lines)
    # error()

    obs_lines = [[[] for _ in Positions()] for _=1:n_particle_value_trains]
    pos_lines = [[[] for _ in Positions()] for _=1:n_particle_value_trains]
    vel_lines = [[[] for _ in Vels()] for _=1:n_particle_value_trains]
    normalized_weight_lines = [[] for _=1:n_particles]
    normalization_line = []

    for (line, newline) in zip(lines, vcat(obs_lines..., pos_lines..., vel_lines..., normalized_weight_lines, [normalization_line]))
        for time in line
            push!(newline, time)
        end
    end


    # println("obs lines:")
    # display(obs_lines)

    # println("pos lines:")
    # display(pos_lines)

    # println("vel lines:")
    # display(vel_lines)

    # TODO: get spiketrains for the internal weight lines

    return (obs_lines, pos_lines, vel_lines, normalized_weight_lines, normalization_line, nothing)
end

function make_one_timestep_figure(gt_tr, inferred_trs; start_caption_letter='a')
    f = Figure(;resolution=(800, 1500))

    particle1_layout = f[1, 1] = GridLayout()
    particle2_layout = f[2, 1] = GridLayout()
    weight_output_layout = f[3, 1] = GridLayout()
    weight_internals_layout = f[4, 1] = GridLayout()

    subcaption_padding = (0, 0, 5, 45)
    subcaption_align = :center

    letters = [start_caption_letter + i for i=0:4]
    Label(f.layout[1, 1, Bottom()], "($(letters[1])) Particle 1 Value Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[2, 1, Bottom()], "($(letters[2])) Particle 2 Value Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[3, 1, Bottom()], "($(letters[3])) Weight-Output Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[4, 1, Bottom()], "($(letters[4])) Internal Weight Term Spiktrains for Particle 1", textsize=17, padding=subcaption_padding, halign=subcaption_align)

    spiketrains = get_spiketrains_for_one_timestep_figure(gt_tr, inferred_trs; n_particle_value_trains=2)

    draw_particle_value_spiketrains!(particle1_layout, spiketrains, 1)
    draw_particle_value_spiketrains!(particle2_layout, spiketrains, 2)
    draw_weight_output_spiketrains!(weight_output_layout, spiketrains)
    draw_weight_internals_spiketrains!(weight_internals_layout, spiketrains, 1)

    f
end

# include("trace_and_particles.jl")

f = make_one_timestep_figure(tr, inferred_trs)