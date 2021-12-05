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

function draw_particle_value_spiketrains!(layout, (obs_lines, pos_lines, vel_lines, _, _, _), particleidx; time_per_step)
    (obsax, posax, velax) = setup_obs_pos_vel_axes(layout)
    colsize!(layout, 1, Relative(0.7))

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
        # ax.yticks = (1:length(varvals), ["$i" for i in varvals])
        ax.ygridvisible = false
        ax.yminorgridvisible = true
        ax.yminorticks = IntervalsBetween(2)
    end
end

function draw_weight_output_spiketrains!(layout, (_, _, _, normalized_weight_lines, normalization_line, _); time_per_step)
    println()
    println("---WEIGHT LINES---")
    display(normalized_weight_lines)
    display(normalization_line)
    draw_weight_spiketrains_2!(layout, (0, time_per_step), normalized_weight_lines, normalization_line)
    colsize!(layout, 1, Relative(0.7))
end
function draw_weight_internals_spiketrains!(f, layout, (_, _, _, _, _, (weightterm_texts, weightterm_lines)), t; time_per_step)
    ax = Axis(layout[1, 1], xlabel="Time (ms)")
    (min_time, max_time) = (0, time_per_step)
    xlims!(ax, (min_time, max_time))

    spikelines = collect(Iterators.flatten((line..., []) for line in weightterm_lines))

    # set up y-ticks to be in the center of the groups of lines corresponding to assemblies.
    # the minor-y-ticks will then be in between these.
    # draw_group_labels! uses these y-ticks as guides to know where the groups are.
    interval_between = length(first(weightterm_lines)) + 1
    ax.yticks=(interval_between/2 + 1):interval_between:length(spikelines)

    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(ax, spikelines, [], [:red for _ in spikelines], 0, min_time, max_time; hide_y_decorations = false)
    ProbEstimates.Spiketrains.SpiketrainViz.draw_group_labels!(
        f, layout, ax,
        [(l, 1) for l in weightterm_texts],
        [:red for _ in spikelines]
    )

    ax.yticksvisible=false
    ax.yticklabelsvisible=false
    ax.ygridvisible=false
    ax.yminorgridvisible = true
    ax.yminorticks = IntervalsBetween(2)
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

    num_neurons_in_assembly_to_show = 5
    weight_term_specs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(1, linespec)
        for linespec in [
            [ProbEstimates.Spiketrains.FwdScoreLine(:xₜ, ProbEstimates.Spiketrains.NeuronInCountAssembly(i)) for i=1:num_neurons_in_assembly_to_show]...,
            [ProbEstimates.Spiketrains.FwdScoreLine(:vₜ, ProbEstimates.Spiketrains.NeuronInCountAssembly(i)) for i=1:num_neurons_in_assembly_to_show]...,
            [ProbEstimates.Spiketrains.FwdScoreLine(:yᵈₜ, ProbEstimates.Spiketrains.NeuronInCountAssembly(i)) for i=1:num_neurons_in_assembly_to_show]...,
            [ProbEstimates.Spiketrains.RecipScoreLine(:xₜ, ProbEstimates.Spiketrains.NeuronInCountAssembly(i)) for i=1:num_neurons_in_assembly_to_show]...,
            [ProbEstimates.Spiketrains.RecipScoreLine(:vₜ, ProbEstimates.Spiketrains.NeuronInCountAssembly(i)) for i=1:num_neurons_in_assembly_to_show]...,
            [ProbEstimates.Spiketrains.RecipScoreLine(:yᵈₜ, ProbEstimates.Spiketrains.NeuronInCountAssembly(i)) for i=1:num_neurons_in_assembly_to_show]...,
        ]
    ]

    weight_term_text_specs = [
        ProbEstimates.Spiketrains.ParticleLineSpec(1, linespec)
        for linespec in [
            ProbEstimates.Spiketrains.FwdScoreText(:xₜ),
            ProbEstimates.Spiketrains.FwdScoreText(:vₜ),
            ProbEstimates.Spiketrains.FwdScoreText(:yᵈₜ),
            ProbEstimates.Spiketrains.RecipScoreText(:xₜ),
            ProbEstimates.Spiketrains.RecipScoreText(:vₜ),
        ]
    ]

    all_specs = vcat(
        obs_particle_linespecs,
        pos_particle_linespecs,
        vel_particle_linespecs,
        normalized_weight_specs,
        [normalization_line_spec],
        weight_term_specs,
        weight_term_text_specs
    )

    # println()
    # println("---SPECS---")
    # display(all_specs[1:20])
    # display(all_specs[21:40])
    # display(all_specs[41:end])
    # println()
    # println()

    # obs_lines = [[[ (;var="obs", particle=i, val=pos)  ] for pos in Positions()] for i=1:n_particle_value_trains]
    # pos_lines = [[[ (;var="pos", particle=i, val=pos) ] for pos in Positions()] for i=1:n_particle_value_trains]
    # vel_lines = [[[ (;var="vel", particle=i, val=vel) ] for vel in Vels()] for i=1:n_particle_value_trains]
    # normalized_weight_lines = [[(var=:normalized_weight, particle=i)] for i=1:n_particles]
    # normalization_line = [(;var="normalization_line")]
    # collectors = vcat(obs_lines..., pos_lines..., vel_lines..., normalized_weight_lines, [normalization_line])

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
    weight_term_lines = [[[] for _=1:num_neurons_in_assembly_to_show] for _=1:6]

    for (line, newline) in zip(lines, vcat(obs_lines..., pos_lines..., vel_lines..., normalized_weight_lines, [normalization_line], weight_term_lines...))
        for time in line
            push!(newline, time)
        end
    end

    weight_term_texts = ["" for _=1:6]
    weight_term_texts[1:5] = lines[end-4:end]
    continuous_term_spikecount = length(weight_term_lines[end])
    continuous_term_estimate = continuous_term_spikecount / ProbEstimates.ContinuousToDiscreteScoreNumSpikes()
    weight_term_texts[end] = "p(yᶜₜ | yᵈₜ)/Q[yᵈₜ ; yᶜₜ] ≈ $continuous_term_estimate"

    # println("obs lines:")
    # display(obs_lines)

    # println("pos lines:")
    # display(pos_lines)

    # println("vel lines:")
    # display(vel_lines)

    println()
    println("---WEIGHT TERM LINES---")
    display(weight_term_texts)
    display(weight_term_lines)
    println()
    println()
    println()

    # TODO: get spiketrains for the internal weight lines

    return (obs_lines, pos_lines, vel_lines, normalized_weight_lines, normalization_line, (weight_term_texts, weight_term_lines))
end

function make_one_timestep_figure(gt_tr, inferred_trs; start_caption_letter='a', time_per_step=150)
    f = Figure(;resolution=(800, 1500))

    particle1_layout = f[1, 1] = GridLayout()
    particle2_layout = f[2, 1] = GridLayout()
    weight_output_layout = f[3, 1] = GridLayout()
    weight_internals_layout = f[4, 1] = GridLayout()

    rowsize!(f.layout, 3, Relative(1/8))

    subcaption_padding = (0, 0, 5, 45)
    subcaption_align = :center

    letters = [start_caption_letter + i for i=0:4]
    Label(f.layout[1, 1, Bottom()], "($(letters[1])) Particle 1 Value Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[2, 1, Bottom()], "($(letters[2])) Particle 2 Value Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[3, 1, Bottom()], "($(letters[3])) Weight-Output Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[4, 1, Bottom()], "($(letters[4])) Internal Weight Term Spiktrains for Particle 1", textsize=17, padding=subcaption_padding, halign=subcaption_align)

    spiketrains = get_spiketrains_for_one_timestep_figure(gt_tr, inferred_trs; n_particle_value_trains=2)

    draw_particle_value_spiketrains!(particle1_layout, spiketrains, 1; time_per_step)
    draw_particle_value_spiketrains!(particle2_layout, spiketrains, 2; time_per_step)
    draw_weight_output_spiketrains!(weight_output_layout, spiketrains; time_per_step)
    draw_weight_internals_spiketrains!(f, weight_internals_layout, spiketrains, 1; time_per_step)

    f
end

# include("trace_and_particles.jl")

f = make_one_timestep_figure(tr, inferred_trs)
