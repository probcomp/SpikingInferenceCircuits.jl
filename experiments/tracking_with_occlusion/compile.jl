using Gen
using Distributions
using DiscreteIRTransforms
using CPTs

includet("model.jl")

latent_domains() = map(EnumeratedDomain, [
    positions(OccluderLength()),
    positions(SquareSideLength()), positions(SquareSideLength()),
    Vels(), Vels()
])

function compile_initial_latents()
    lcpts = to_labeled_cpts(
        initial_latents,
        (EnumeratedDomain([nothing]),)
    )

    (with_cpts, dom_maps) = to_indexed_cpts(
        initial_latents,
        (EnumeratedDomain([nothing]),)
    )
    return (with_lcpts, with_cpts, dom_maps)
end
function compile_step()
    lcpts = to_labeled_cpts(step, latent_domains())
    (cpts, dom_maps) = to_indexed_cpts(step, latent_domains())
    return (lcpts, cpts, dom_maps)
end
function compile_obs()
    lcpts = to_labeled_cpts(observation, latent_domains())
    # (cpts, dom_maps) = to_indexed_cpts(observation, latent_domains())
    # return (lcpts, cpts, dom_maps)
    return lcpts
end

function compile_obs1d()
    lcpts = to_labeled_cpts(obs_1d, map(EnumeratedDomain, (positions(OccluderLength()), positions(SquareSideLength()))))
    (cpts, dom_maps) = to_indexed_cpts(obs_1d, map(EnumeratedDomain, (positions(OccluderLength()), positions(SquareSideLength()))))
    return (lcpts, cpts, dom_maps)
    return lcpts
end


(lcpts, lcpts, dom_maps) = compile_obs1d() #compile_obs()
# lcpts = compile_obs()
@load_generated_functions()

#=
 
=#