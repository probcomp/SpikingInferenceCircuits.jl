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
# include("pm_model.jl")
include("../inference.jl")
include("../visualize.jl")
ProbEstimates.DoRecipPECheck() = false
include("../utils.jl")

make_exact_bayes_filter_heatmaps!(layout, gt_tr) =
    draw_2d_posterior!(layout, get_enumeration_grids(gt_tr); show_statistics=false)

function make_layout(f, fpos)
    f[fpos...] = GridLayout()
end
#=
inferred_trs is a vector [[(tr, log_importance_weight) for _=1:n_particles] for _=1:n_timesteps]
The first 2 particles at each timestep are "distinguished".
=#
function make_figure(gt_tr, inferred_trs)
    f = Figure()
    
    make_exact_bayes_filter_heatmaps!(make_layout(f, (1, 1)), gt_tr)
    # draw_particles_visualization!(make_layout(f, (2, 1)), inferred_trs)
    # draw_value_spiketrains!(make_layout(f, (3, 1)), inferred_trs)
    # draw_score_spiketrains!(make_layout(f, (4, 1)), inferred_trs)

    f
end
make_figure(gt_tr; n_particles=10) = make_figure(gt_tr, 
    smc(tr, n_particles, exact_init_proposal, exact_step_proposal;
        ess_threshold = -Inf # no resampling
    )
)
make_figure(; n_particles=10, n_steps=10) = make_figure(generate(model, (n_steps,))[1]; n_particles)