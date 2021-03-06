using DynamicModels
include("model.jl")
include("groundtruth_rendering.jl")
include("prior_proposal.jl")
include("visualize.jl")
include("locally_optimal_proposal.jl")

use_ngf() = true

if use_ngf()
    ProbEstimates.use_noisy_weights!()
else
    ProbEstimates.use_perfect_weights!()
end

model = @DynamicModel(init_latent_model, step_latent_model, obs_model, 5)
init_prop = @compile_initial_proposal(_init_proposal, 1)
step_prop = @compile_step_proposal(_step_proposal, 5, 1)

@load_generated_functions()

obs_choicemap_to_vec_of_vec(ch) = [
    [
        ch[:img_inner => x => y => :pixel_color => :val]
        for x=1:ImageSideLength()
    ]
    for y=1:ImageSideLength()
]

### Run inference:
do_inference(gt_tr; n_particles=10) = dynamic_model_smc(
    model, get_dynamic_model_obs(gt_tr),
    cm -> (obs_choicemap_to_vec_of_vec(cm),),
    init_prop, step_prop, n_particles
);

function make_gt_particle_viz(gt_tr, unweighted_inferred_trs)
    GLMakie.activate!()
    nparticles = length(first(unweighted_inferred_trs))
    draw_gt_and_particles(gt_tr, unweighted_inferred_trs,
    "$nparticles-particle SMC w/ locally-optimal proposal. Run in $(use_ngf() ? "NeuralGen-Fast." : "Vanilla Gen.")"
    );
end
function make_gt_particle_viz_img_only(gt_tr, unweighted_inferred_trs)
    GLMakie.activate!()
    nparticles = length(first(unweighted_inferred_trs))
    draw_gt_particles_img_only(gt_tr, unweighted_inferred_trs,
    "$nparticles-particle SMC w/ locally-optimal proposal. Run in $(use_ngf() ? "NeuralGen-Fast." : "Vanilla Gen.")"
    );
end

function surround3(ch, a, dom)
    v = ch[a => :val]
    if v-1 in dom && v+1 in dom
        return (v-1):v+1
    elseif v-1 in dom && v-2 in dom
        return (v-2):v
    else
        return v:(v+2)
    end
end
latent_domains_for_viz(ch)     = (
    occₜ = surround3(ch, :occₜ, positions(OccluderLength())),
    xₜ   = surround3(ch, :xₜ, positions(SquareSideLength())),
    yₜ   = surround3(ch, :yₜ, positions(SquareSideLength())),
    vxₜ  = surround3(ch, :vxₜ, Vels()),
    vyₜ  = surround3(ch, :vyₜ, Vels())
)

function make_spiketrain_fig(tr, neurons_to_show_indices=1:3; nest_all_at, kwargs...)
    ProbEstimates.Spiketrains.SpiketrainViz.CairoMakie.activate!()
    propose_sampling_tree = Dict(
        :occₜ => [], :xₜ => [:occₜ], :yₜ => [],
        :vxₜ => [:xₜ], :vyₜ => [:yₜ]
    )
    assess_sampling_tree = Dict(
        :occₜ => [], :vxₜ => [], :vyₜ => [],
        :xₜ => [:occₜ, :vxₜ],
        :yₜ => [:vyₜ]
    )
    propose_addr_topological_order = [:occₜ, :xₜ, :yₜ, :vxₜ, :vyₜ]
    
    doms = latent_domains_for_viz(get_submap(get_choices(tr), nest_all_at))
    return ProbEstimates.Spiketrains.draw_spiketrain_group_fig(
        ProbEstimates.Spiketrains.value_neuron_scores_groups(keys(doms), values(doms), neurons_to_show_indices), tr,
        (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order);
        nest_all_at, kwargs...
    )
end


### Generate a particular trace:
occluded_bounce_constraints() = choicemap(
	(:init => :latents => :xₜ => :val, 3),
    (:init => :latents => :yₜ => :val, 9),
	(:init => :latents => :vxₜ => :val, 2),
    (:init => :latents => :vyₜ => :val, -1),
    (:init => :latents => :occₜ => :val, 8),
    (:steps => 1 => :latents => :xₜ => :val, 5),
    (:steps => 1 => :latents => :yₜ => :val, 8),
    (:steps => 1 => :latents => :occₜ => :val, 8),
    (:steps => 2 => :latents => :xₜ => :val, 7),
    (:steps => 2 => :latents => :yₜ => :val, 7),
    (:steps => 2 => :latents => :occₜ => :val, 8),
    (:steps => 3 => :latents => :occₜ => :val, 8),
    (:steps => 4 => :latents => :occₜ => :val, 8)
)

generate_occluded_bounce_tr() = generate(model, (15,), occluded_bounce_constraints())[1]

## Script to run inference + make visualizations
VelOneOffProb() = 0.1
gt_tr = generate_occluded_bounce_tr();
VelOneOffProb() = 0.2
(unweighted_trs, _) = do_inference(gt_tr; n_particles=30);

# # Inference results animation:

(fig, t) = make_gt_particle_viz_img_only(gt_tr, unweighted_trs); t[] = 4; fig
t[] = 2
save("inferenceframe1.png", fig)
t[] = 4
save("inferenceframe2.png", fig)
t[] = 6
save("inferenceframe3.png", fig)

(obsfig, tobs) = draw_obs(gt_tr)
tobs[] = 6
save("obs.png", obsfig)

# # Spiketrain figure:
# f = make_spiketrain_fig(
#     last(unweighted_trs)[1], 1:3; nest_all_at=(:steps => 1 => :latents),
#     resolution=(600, 450), figure_title="Dynamically Weighted Spike Code from Inference"
# )

# Draw observations:
# tr, _= generate(model, (2,), choicemap(
#     (:init => :latents => :xₜ => :val, 6),
#     (:init => :latents => :yₜ => :val, 6),
#     (:init => :latents => :occₜ => :val, 7)
# ))
# using CairoMakie
# CairoMakie.activate!()
# (fig, t) = draw_obs(tr); fig

