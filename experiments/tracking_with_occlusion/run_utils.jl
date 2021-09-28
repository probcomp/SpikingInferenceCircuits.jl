function get_returned_obs(gt_tr)
    firstcm, restcms = get_dynamic_model_obs(gt_tr)
    selection = select((
        :img_inner => x => y => :pixel_color => :color
        for x in positions(SquareSideLength()) for y in positions(SquareSideLength())
    )...)
    filtercm(cm) = get_selected(cm, selection)
    return (filtercm(firstcm), map(filtercm, restcms))
end

obs_choicemap_to_vec_of_vec(ch) = [
    [
        ch[:img_inner => x => y => :pixel_color => :color => :val]
        for y=1:ImageSideLength()
    ]
    for x=1:ImageSideLength()
]

### Inference visualization:
function make_gt_particle_viz(gt_tr, unweighted_inferred_trs)
    GLMakie.activate!()
    nparticles = length(first(unweighted_inferred_trs))
    draw_gt_and_particles(gt_tr, unweighted_inferred_trs,
    "$nparticles-particle SMC w/ nearly locally-optimal proposal. Run in $(use_ngf() ? "NeuralGen-Fast." : "Vanilla Gen.")"
    );
end
function make_gt_particle_viz_img_only(gt_tr, unweighted_inferred_trs)
    GLMakie.activate!()
    nparticles = length(first(unweighted_inferred_trs))
    draw_gt_particles_img_only(gt_tr, unweighted_inferred_trs,
    "$nparticles-particle SMC w/ nearly locally-optimal proposal. Run in $(use_ngf() ? "NeuralGen-Fast." : "Vanilla Gen.")"
    );
end

### Spiketrain visualization:
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
    (:steps => 1 => :latents => :vxₜ => :val, 2),
    (:steps => 1 => :latents => :vyₜ => :val, -1),
    (:steps => 1 => :latents => :xₜ => :val, 5),
    (:steps => 1 => :latents => :yₜ => :val, 8),
    (:steps => 1 => :latents => :occₜ => :val, 8),
    (:steps => 2 => :latents => :vxₜ => :val, 2),
    (:steps => 2 => :latents => :vyₜ => :val, -1),
    (:steps => 2 => :latents => :xₜ => :val, 7),
    (:steps => 2 => :latents => :yₜ => :val, 7),
    (:steps => 2 => :latents => :occₜ => :val, 8),
    (:steps => 3 => :latents => :occₜ => :val, 8),
    (:steps => 4 => :latents => :occₜ => :val, 8),
    (:steps => 5 => :latents => :occₜ => :val, 8),
    (:steps => 6 => :latents => :occₜ => :val, 8),
    (:steps => 7 => :latents => :occₜ => :val, 8)
)

generate_occluded_bounce_tr() = generate(model, (15,), occluded_bounce_constraints())[1]
