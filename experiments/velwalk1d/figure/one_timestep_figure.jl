include("shared.jl")
include("../model_continuous_obs.jl")
include("../proposals_continuous_obs.jl")

function draw_particle_value_spiketrains!(layout, (obs_lines, pos_lines, vel_lines, _, _, _), particleidx; time_per_step)
    (obsax, posax, velax) = setup_obs_pos_vel_axes(layout)
    colsize!(layout, 1, Relative(0.7))

    velax.xlabel = "Time (ms)"
    velax.xlabelpadding = -5.0

    (min_time, max_time) = (0, 2*time_per_step)
    xlims!(obsax, (min_time, max_time))
    xlims!(posax, (min_time, max_time))
    xlims!(velax, (min_time, max_time))

    color = (particleidx == 1) ? :red : (particleidx == 2) ? :blue : :black
    poscolors = [color for _ in Positions()]
    velcolors = [color for _ in Vels()]

    # println("particle index = $particleidx; OBS LINES:")
    # display(obs_lines)

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
    # println()
    # println("---WEIGHT LINES---")
    # display(normalized_weight_lines)
    # display(normalization_line)
    draw_weight_spiketrains_2!(layout, (0, 2*time_per_step), normalized_weight_lines, normalization_line)
    colsize!(layout, 1, Relative(0.7))
end
function draw_weight_internals_spiketrains!(f, layout, (_, _, _, _, _, (weightterm_texts, weightterm_lines)), t; time_per_step)
    ax = Axis(layout[1, 1], xlabel="Time (ms)")
    (min_time, max_time) = (0, 2*time_per_step)
    xlims!(ax, (min_time, max_time))
    ax.xlabelpadding = -5

    spikelines = collect(Iterators.flatten((line[1:3]..., [], [], []) for line in weightterm_lines))

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

function get_spiketrains_for_one_timestep_figure(
    gt_tr, inferred_trs;
    num_autonormalization_spikes=nothing,
    n_particle_value_trains, time_per_step
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

    obs_lines = [[[] for _ in Positions()] for _=1:n_particle_value_trains]
    pos_lines = [[[] for _ in Positions()] for _=1:n_particle_value_trains]
    vel_lines = [[[] for _ in Vels()] for _=1:n_particle_value_trains]
    normalized_weight_lines = [[] for _=1:n_particles]
    normalization_line = []
    weight_term_lines = [[[] for _=1:num_neurons_in_assembly_to_show] for _=1:6]

    lines = nothing
    for t_plus_1=2:3
        particles = inferred_trs[t_plus_1]

        previous_logweights = map(x -> x[2], inferred_trs[t_plus_1 - 1])
        previous_normalized_logweights = previous_logweights .- logsumexp(previous_logweights)
        current_logweights = map(x -> x[2], inferred_trs[t_plus_1])
        log_weight_updates = current_logweights .- previous_logweights

        lines = ProbEstimates.Spiketrains.get_lines_for_multiparticle_specs(
            all_specs,
            [tr for (tr, wt) in particles], # traces
            log_weight_updates, # log_weight_updates which should be computed
            spiketrain_data_args;
            nest_all_at=nestat_addr(t_plus_1 - 1),
            other_factors_to_multiply_in=exp.(previous_normalized_logweights),
            num_autonormalization_spikes,
            vars_disc_to_cont=Dict(:yᵈₜ => (nest_addr -> (ProbEstimates.Spiketrains.nest(to_obs_nest_addr(nest_addr), :yᶜₜ))))
        )
    
        for (line, newline) in zip(lines, vcat(obs_lines..., pos_lines..., vel_lines..., normalized_weight_lines, [normalization_line], weight_term_lines...))
            for time in line
                push!(newline, time + (t_plus_1 - 2)*time_per_step)
            end
        end    
    end

    weight_term_texts = ["" for _=1:6]
    weight_term_texts[1:5] = lines[end-4:end]
    continuous_term_spikecount = length(weight_term_lines[end])
    continuous_term_estimate = continuous_term_spikecount / ProbEstimates.ContinuousToDiscreteScoreNumSpikes()
    weight_term_texts[end] = "p(yᶜₜ | yᵈₜ)/Q[yᵈₜ ; yᶜₜ] ≈ $continuous_term_estimate"

    # previous_logweights = map(x -> x[2], inferred_trs[T])
    # previous_normalized_logweights = previous_logweights .- logsumexp(previous_logweights)
    # current_logweights = map(x -> x[2], inferred_trs[T + 1])
    # log_weight_updates = current_logweights .- previous_logweights
    # lines = ProbEstimates.Spiketrains.get_lines_for_multiparticle_specs(
    #     all_specs,
    #     [tr for (tr, wt) in last(inferred_trs)], # traces
    #     log_weight_updates, # log_weight_updates which should be computed
    #     spiketrain_data_args;
    #     nest_all_at=nestat_addr(T),
    #     other_factors_to_multiply_in=exp.(previous_normalized_logweights),
    #     num_autonormalization_spikes,
    #     vars_disc_to_cont=Dict(:yᵈₜ => (nest_addr -> (ProbEstimates.Spiketrains.nest(to_obs_nest_addr(nest_addr), :yᶜₜ))))
    # )

    # println()
    # println()
    # println("---LINES---")
    # display(lines[1:20])
    # display(lines[21:40])
    # display(lines[41:end])
    # println()

    # lines = ProbEstimates.Spiketrains.get_lines_for_multiparticle_specs(
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

    # println("obs lines:")
    # display(obs_lines)

    # println("pos lines:")
    # display(pos_lines)

    # println("vel lines:")
    # display(vel_lines)

    # println()
    # println("---WEIGHT TERM LINES---")
    # display(weight_term_texts)
    # display(weight_term_lines)
    # println()
    # println()
    # println()

    # TODO: get spiketrains for the internal weight lines

    return (obs_lines, pos_lines, vel_lines, normalized_weight_lines, normalization_line, (weight_term_texts, weight_term_lines))
end

function draw_assembly_spikes!(layout, gt_tr, ax_to_link_to; maxtime)
    ax = Axis(layout[1, 1], xlabel="Time (ms)")
    linkyaxes!(ax_to_link_to, ax)
    ax.ygridvisible = false
    ax.yticksvisible=false
    ax.xlabelpadding = -5.0

    contval = gt_tr[:steps => 2 => :obs => :yᶜₜ => :val]
    excitations = [exp(Gen.logpdf(normal, contval, discval, Yᶜ_STD())) for discval in Positions()]

    trains = [[] for _ in Positions()]
    num_spikes = poisson(sum(excitations) * ProbEstimates.ContinuousToDiscreteScoreNumSpikes())
    for _=1:num_spikes
        idx = categorical(normalize(excitations))
        time = uniform_discrete(0, ProbEstimates.Latency())
        push!(trains[idx], time)
    end
    for train in trains
        sort!(train)
    end

    ProbEstimates.Spiketrains.SpiketrainViz.draw_lines!(
        ax, reverse(trains), [], [:black for train in trains], 0, 0, ProbEstimates.Latency() + 10;
        hide_y_decorations = false
    )
end

function draw_obs_particles_2steps!(layout, tr, inferred_trs)
    inferred_trs = inferred_trs[end-1:end]

    axislayout = layout[1, 1] = GridLayout()
    (velax, posax) = setup_vel_pos_axes(axislayout)
    n_particles = length(first(inferred_trs))

    colors = [:red, :blue, (:black for _=3:n_particles)...]
    # draw inferred trs
    # (pos_particles, vel_particles) = get_particle_weights_colors(inferred_trs)
    # particles = draw_particle_squares_for_variable!(posax, Positions(), pos_particles, n_particles; starttime=1, T=2)
    # draw_particle_squares_for_variable!(velax, Vels(), vel_particles, n_particles; starttime=1, T=2)    
    particles = []
    for t=(1:get_args(tr)[1])
        for (i, (color, (particle, _))) in enumerate(zip(colors, inferred_trs[t]))
            vel = latents_choicemap(particle, t)[:vₜ => :val]
            pos = latents_choicemap(particle, t)[:xₜ => :val]

            c = n_particles/2
            scatter!(velax, [Point2(t + (i - c)/50, vel)]; color, markersize=12)
            push!(particles, scatter!(posax, [Point2(t + (i - c)/50, pos)]; color, markersize=12))
        end
    end

    # draw obs
    times = 1:(get_args(tr)[1])
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

    # xlims!(posax, (1., 3.))
    # xlims!(velax, (1., 3.))
    posax.xticks = ([1, 2], ["t-1", "t"])
end

function draw_tuning_curves!(layout, gt_tr, inferred_trs)
    ax = Axis(layout[1, 1], ylabel="yᵈₜ", xlabel="pdf(yᶜₜ | yᵈₜ)")
    ax.xlabelpadding = -5.0
    ax.yticks=Positions()
    yvals = (first(Positions()) - 1):0.01:(last(Positions()) + 1)
    density(xᵈ, σ) = [exp(Gen.logpdf(normal, c, xᵈ, σ)) for c in (first(Positions()) - 1):0.01:(last(Positions()) + 1)]
    density_points(xᵈ, σ) = [Point2(d + 1, i) for (i, d) in zip(yvals, density(xᵈ, σ))]

    maxval = -Inf
    trueval = gt_tr[:steps => 2 => :obs => :yᶜₜ => :val]
    for i in Positions()
        dens = density(i, Yᶜ_STD())
        maxval = max(maxval, maximum(dens))
        lines!(ax, density_points(i, Yᶜ_STD()), color=RGBA(0, 0, 0, min(1, 0.1 + exp(Gen.logpdf(normal, trueval, i, Yᶜ_STD())/5))))
    end
    ax.xticks=[0, ceil(maxval) + 1]

    ax.ygridvisible = false
    ax.xgridvisible=false
    ax.topspinevisible = false
    ax.rightspinevisible=false

    scatter!(ax, [Point2(0, trueval)], color=:seagreen, markersize=12, marker='■')
    for (i, ((tr, _), color)) in enumerate(Iterators.reverse(zip(inferred_trs[3], [:red, :blue, [:black for _=3:length(inferred_trs[3])]...])))
        scatter!(ax, [Point2((i + 1)/5, tr[:steps => 2 => :latents => :yᵈₜ => :val])]; color, markersize=12)
    end
    return ax
end

function make_one_timestep_figure(gt_tr, inferred_trs; start_caption_letter='b', time_per_step=200)
    f = Figure(;resolution=(800, 1250))

    # lhs_layout = f[1, 1] = GridLayout()
    # rhs_layout = f[1, 2] = GridLayout()

    weight_output_layout = f.layout[1, 1] = GridLayout()
    weight_internals_layout = f.layout[2, 1] = GridLayout()
    particle1_layout = f.layout[3, 1] = GridLayout()
    particle2_layout = f.layout[4, 1] = GridLayout()

    # rowsize!(f.layout, 3, Relative(1/8))
    # colsize!(f.layout, 2, Relative(800/2000))

    # toprow_layout = f[1, 1] = GridLayout()
    # transition_layout = toprow_layout[1, 1] = GridLayout()
    # tuningcurve_layout = toprow_layout[1, 2] = GridLayout()
    # qassembly_layout = toprow_layout[1, 3] = GridLayout()
    # colsize!(toprow_layout, 1, Relative(2/5))
    # colsize!(toprow_layout, 2, Relative(1/10))
    # draw_obs_particles_2steps!(transition_layout, gt_tr, inferred_trs)
    # ax = draw_tuning_curves!(tuningcurve_layout, gt_tr, inferred_trs)
    # draw_assembly_spikes!(qassembly_layout, gt_tr, ax; maxtime=time_per_step)

    subcaption_padding = (0, 0, 5, 45)
    subcaption_align = :center
    # Label(toprow_layout[1, 1, Bottom()], "($start_caption_letter) Observation and Particle Transition", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    # Label(toprow_layout[1, 2, Bottom()], "($(start_caption_letter+1)) Tuning Curves", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    # Label(toprow_layout[1, 3, Bottom()], "($(start_caption_letter + 2)) Spikes from assembly for Q[yᵈₜ ; yᶜₜ]", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    # start_caption_letter += 3

    letters = [start_caption_letter + i for i=0:4]
    Label(f.layout[1, 1, Bottom()], "($(letters[1])) Weight-Output Assembly-Level Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[2, 1, Bottom()], "($(letters[2])) Internal Weight Term Spiketrains for Particle 1", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[3, 1, Bottom()], "($(letters[3])) Particle 1 Value Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)
    Label(f.layout[4, 1, Bottom()], "($(letters[4])) Particle 2 Value Spiketrains", textsize=17, padding=subcaption_padding, halign=subcaption_align)

    spiketrains = get_spiketrains_for_one_timestep_figure(gt_tr, inferred_trs; n_particle_value_trains=2, time_per_step)

    draw_particle_value_spiketrains!(particle1_layout, spiketrains, 1; time_per_step)
    draw_particle_value_spiketrains!(particle2_layout, spiketrains, 2; time_per_step)
    draw_weight_output_spiketrains!(weight_output_layout, spiketrains; time_per_step)
    draw_weight_internals_spiketrains!(f, weight_internals_layout, spiketrains, 1; time_per_step)

    (f, ax)
end

include("trace_and_particles.jl")

(f, ax) = make_one_timestep_figure(tr, inferred_trs)
f
