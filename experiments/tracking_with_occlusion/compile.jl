using Gen
using Distributions
using DiscreteIRTransforms
using CPTs

includet("model.jl")

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
    arg_domains = map(EnumeratedDomain, [
        positions(OccluderLength()),
        positions(SquareSideLength()), positions(SquareSideLength()),
        Vels(), Vels()
    ])

    lcpts = to_labeled_cpts(step, arg_domains)
    (cpts, dom_maps) = to_indexed_cpts(step, arg_domains)
    return (lcpts, cpts, dom_maps)
end

(lcpts, cpts, dom_maps) = compile_step()
@load_generated_functions()

#=
 
=#