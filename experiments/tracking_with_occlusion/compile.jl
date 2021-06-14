using Gen
using DiscreteIRTransforms
using CPTs

includet("model.jl")

lcpts = to_labeled_cpts(
    initial_latents,
    (EnumeratedDomain([nothing]),)
)

(with_cpts, dom_maps) = to_indexed_cpts(
    initial_latents,
   (EnumeratedDomain([nothing]),)
)

@load_generated_functions()

#=
 map(EnumeratedDomain, [
        positions(OccluderLength()),
        positions(SquareSideLength()), positions(SquareSideLength()),
        Vels(), Vels()
    ])
=#