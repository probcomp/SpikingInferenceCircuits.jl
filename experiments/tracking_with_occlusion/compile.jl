using Gen
using Distributions
using DiscreteIRTransforms
using CPTs
using SpikingInferenceCircuits
using Revise

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
    return (lcpts, with_cpts, dom_maps)
end
function compile_step()
    lcpts = to_labeled_cpts(step, latent_domains())
    (cpts, dom_maps) = to_indexed_cpts(step, latent_domains())
    return (lcpts, cpts, dom_maps)
end
function compile_obs()
    lcpts = to_labeled_cpts(observation, latent_domains())
    (cpts, dom_maps) = to_indexed_cpts(observation, latent_domains())
    return (lcpts, cpts, dom_maps)
    # return lcpts
end

latent_doms_1d() = map(EnumeratedDomain, (positions(OccluderLength()), positions(SquareSideLength())))
function compile_obs1d()
    lcpts = to_labeled_cpts(obs_1d, latent_doms_1d())
    (cpts, dom_maps) = to_indexed_cpts(obs_1d, map(EnumeratedDomain, (positions(OccluderLength()), positions(SquareSideLength()))))
    return (lcpts, cpts, dom_maps)
    # return lcpts
end


(lcpts, icpts, dom_maps) = compile_obs1d()
# lcpts = compile_obs1d()
println("Compiled 1D obs model.")

@load_generated_functions()
println("Loaded generated functions.")

# ### Circuit compilation
circuit = gen_fn_circuit(
    icpts,
    map(d -> FiniteDomain(length(DiscreteIRTransforms.vals(d))), latent_doms_1d()),
    Assess()
)
println("Circuit constructed.")

includet("../neurips_tracking/implementation_rules.jl")
println("Implemenation rules loaded.")

impl = SpikingInferenceCircuits.Circuits.memoized_implement_deep(
    circuit,
    SpikingInferenceCircuits.Spiking()
);

println("Circuit implemented deeply.")


# lcpts = to_labeled_cpts(render_pixel, latent_domains())
# icpts, maps = to_indexed_cpts(render_pixel, map(EnumeratedDomain, (
#     positions(OccluderLength()), (positions(SquareSideLength()) for _=1:4)...
# )))
# @load_generated_functions