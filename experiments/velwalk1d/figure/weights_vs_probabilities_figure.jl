include("shared.jl")

function draw_obs_particles!(layout, tr, inferred_trs)
    axislayout = layout[1, 1] = GridLayout()
    (velax, posax) = setup_vel_pos_axes(axislayout)
    n_particles = length(first(inferred_trs))

    xlims!(posax, (0., 3.))
    xlims!(velax, (0., 3.))

    # draw inferred trs
    (pos_particles, vel_particles) = get_particle_weights_colors(inferred_trs)
    particles = draw_particle_squares_for_variable!(posax, Positions(), pos_particles, n_particles)
    println(particles)
    draw_particle_squares_for_variable!(velax, Vels(), vel_particles, n_particles)    

    # draw obs
    times = 0:(get_args(tr)[1])
    pos_observations = [obs_choicemap(tr, t)[:yᵈₜ => :val] for t in times]
    obs = scatter!(posax, times, pos_observations, color=:seagreen, markersize=20, marker='■')

    l = Legend(layout[2, 1], [obs, particles], ["Observed Positions", "Inferred Particles"], orientation=:horizontal)
    l.tellwidth=false
    l.tellheight=true
    rowgap!(layout, 10)
end

function draw_weight_spiketrains!(layout, inferred_trs; time_per_step=200, num_autonormalization_spikes_for_each_timestep)
    n_particles = length(first(inferred_trs))
    example_tr_at_end = inferred_trs[end][1][1]
    T = get_args(example_tr_at_end)[1]
    ms_timerange = (0, time_per_step * (T + 1))
    (pos_lines, vel_lines, normalized_weight_lines, normalization_line) =
        get_pos_vel_value_spiketrains(time_per_step, inferred_trs; num_autonormalization_spikes_for_each_timestep);
    
    println("got spiketrains")
    
    return (
        draw_weight_spiketrains_2!(layout, ms_timerange, normalized_weight_lines, normalization_line),
        normalized_weight_lines, normalization_line
    )
end
function draw_probability_lines!(layout, inferred_trs, weight_spiketrains, axis_to_link_to; time_per_step=200)
    ax = Axis(layout[1, 1], xlabel="Time (ms)", ylabel="Particle Probability")
    linkxaxes!(axis_to_link_to, ax)
    ylims!(ax, (0., 1.))

    # weight(t, particle_idx) = exp(inferred_trs[t + 1][particle_idx][2])
    # use spiketrains for weights, so that we include the noise from readout after auto-normalization
    weight(t, particle_idx) = length([x for x in weight_spiketrains[particle_idx] if t * time_per_step ≤ x ≤ (t+1)*time_per_step])
    weights = [[weight(t, particle_idx) for particle_idx=1:length(inferred_trs)] for t=0:2]
    probs = [ w / sum(w) for w in weights ]

    lines(particle_idx) = [
        [
            Point2f0(t * time_per_step, probs[t+1][particle_idx]),
            Point2f0((t + 1) * time_per_step, probs[t+1][particle_idx])
        ]
        for t=0:2
    ] |> Iterators.flatten |> collect

    lines!(ax, lines(1); color=:blue)
    lines!(ax, lines(2); color=:red)
    for i=3:length(inferred_trs)
        lines!(ax, lines(i); color=:black)
    end
end
function draw_weight_spiketrains_and_lines!(layout, inferred_trs; time_per_step=200, num_autonormalization_spikes_for_each_timestep)
    spiketrain_layout = layout[1, 1] = GridLayout()
    weight_line_layout = layout[2, 1] = GridLayout()

    ((weight_ax, _), normalized_weight_lines, _) =
        draw_weight_spiketrains!(spiketrain_layout, inferred_trs; time_per_step, num_autonormalization_spikes_for_each_timestep)
    draw_probability_lines!(weight_line_layout, inferred_trs, normalized_weight_lines, weight_ax; time_per_step)
end

function make_figure(tr, inferred_trs, num_autonormalization_spikes_for_each_timestep=[nothing for _ in inferred_trs])
    f = Figure(; resolution=(600, 800))
    l1 = f[1, 1] = GridLayout()
    l2 = f[2, 1] = GridLayout()

    subcaption_padding = (0, 0, 5, 45)
    subcaption_align = :center
    Label(f.layout[1, 1, Bottom()], "(a) Observed transition & Particle Values", textsize=17, padding=(0, 0, 5, 15), halign=subcaption_align)
    Label(f.layout[2, 1, Bottom()], "(b) Particle weight spiketrains & normalized probability values", textsize=17, padding=(0, 0, 5, 50), halign=subcaption_align)
    

    draw_weight_spiketrains_and_lines!(l2, inferred_trs; num_autonormalization_spikes_for_each_timestep)
    draw_obs_particles!(l1, tr, inferred_trs)

    f
end

GLMakie.activate!()

tr = generate(model, (2,), choicemap(
    (:init => :latents => :xₜ => :val, 5),
    (:init => :latents => :vₜ => :val, 0),
    (:steps => 1 => :latents => :xₜ => :val, 5),
    (:steps => 1 => :latents => :vₜ => :val, 0),
    (:steps => 2 => :latents => :xₜ => :val, 5),
    (:steps => 2 => :latents => :vₜ => :val, 0),

    (:init => :obs => :yᵈₜ => :val, 4),
    (:steps => 1 => :obs => :yᵈₜ => :val, 5),
    (:steps => 2 => :obs => :yᵈₜ => :val, 5),
))[1]

inferred_trs = predetermined_smc(tr, 3, exact_init_proposal, approx_step_proposal,
    (
        [
            choicemap((:init => :latents => :xₜ => :val, 5), (:init => :latents => :vₜ => :val, -1)),
            choicemap((:init => :latents => :xₜ => :val, 5), (:init => :latents => :vₜ => :val, 0)),
            choicemap((:init => :latents => :xₜ => :val, 5), (:init => :latents => :vₜ => :val, 1))
        ],
        [
            [
                choicemap((:steps => 1 => :latents => :xₜ => :val, 4), (:steps => 1 => :latents => :vₜ => :val, -1)),
                choicemap((:steps => 1 => :latents => :xₜ => :val, 5), (:steps => 1 => :latents => :vₜ => :val, 0)),
                choicemap((:steps => 1 => :latents => :xₜ => :val, 6), (:steps => 1 => :latents => :vₜ => :val, 1))
            ],
            [
                choicemap((:steps => 2 => :latents => :xₜ => :val, 3), (:steps => 2 => :latents => :vₜ => :val, -1)),
                choicemap((:steps => 2 => :latents => :xₜ => :val, 5), (:steps => 2 => :latents => :vₜ => :val, 0)),
                choicemap((:steps => 2 => :latents => :xₜ => :val, 6), (:steps => 2 => :latents => :vₜ => :val, 0))
            ]
        ]
    );
    ess_threshold=-Inf # no resampling
)[2];
nspikes = [1,1,1];

f = make_figure(tr, inferred_trs, nspikes)

#=
TODOs:
- generate specific initial trace
- code to produce specific inferred traces 
- produce inferred traces which look like they will exhibit the right effect

- "function call" to force the auto-normalization circuit to produce a
specific number of auto-normalization spikes which will exhibit the effect
we care about

- implement function call
- debugging
=#

f