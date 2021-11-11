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
    # draw_value_spiketrains!(make_layout(f, (3, 1)), inferred_trs)
    # draw_score_spiketrains!(make_layout(f, (4, 1)), inferred_trs)

    f
end
make_figure(gt_tr; n_particles=10) = make_figure(gt_tr, 
    smc(gt_tr, n_particles, exact_init_proposal, exact_step_proposal;
        ess_threshold = -Inf # no resampling
    )[2]
)
make_figure(; n_particles=10, n_steps=6) = make_figure(generate(model, (n_steps,))[1]; n_particles)

make_figure()