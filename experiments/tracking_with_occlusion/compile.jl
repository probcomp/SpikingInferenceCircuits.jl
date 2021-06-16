using Gen
using Distributions
using DiscreteIRTransforms
using CPTs
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits
using SpikingCircuits
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

consts_compiled = DiscreteIRTransforms.inline_constant_nodes(icpts)
@load_generated_functions()

println("Loaded generated functions.")

tr = simulate(consts_compiled, (2,2));

pix_renderer = DiscreteIRTransforms.get_ir(consts_compiled).nodes[5].generative_function.kernels[1]

# ### Circuit compilation
circuit = gen_fn_circuit(
    pix_renderer,
    map(d -> FiniteDomain(length(DiscreteIRTransforms.vals(d))), latent_doms_1d()),
    Assess()
)
println("Circuit constructed.")

includet("../neurips_tracking/implementation_rules.jl")
println("Implemenation rules loaded.")

impl1 = SpikingInferenceCircuits.Circuits.implement(
    circuit,
    SpikingInferenceCircuits.Spiking()
);
println("Circuit implemented once.")
impl2 = SpikingInferenceCircuits.Circuits.implement(
    impl1,
    SpikingInferenceCircuits.Spiking()
);
println("Circuit implemented twice.")
impl3 = SpikingInferenceCircuits.Circuits.implement(
    impl1,
    SpikingInferenceCircuits.Spiking()
);
println("Circuit implemented thrice.")

impl_deep = SpikingInferenceCircuits.Circuits.implement_deep(
    impl3,
    SpikingInferenceCircuits.Spiking()
);
println("Circuit implemented deeply.")


# println("Circuit implemented deeply.")


# lcpts = to_labeled_cpts(render_pixel, latent_domains())
# icpts, maps = to_indexed_cpts(render_pixel, map(EnumeratedDomain, (
#     positions(OccluderLength()), (positions(SquareSideLength()) for _=1:4)...
# )))
# @load_generated_functions