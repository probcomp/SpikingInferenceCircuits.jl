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

function compile_obs()
    lcpts = to_labeled_cpts(observation, latent_domains())
    (cpts, dom_maps) = to_indexed_cpts(observation, latent_domains())
    return (lcpts, cpts, dom_maps)
    # return lcpts
end

(lcpts, icpts, dom_maps) = compile_obs()

println("Compiled 2D obs model to indexed CPTS.")

@load_generated_functions()

consts_compiled = DiscreteIRTransforms.inline_constant_nodes(icpts)
println("Inlined constants.")
@load_generated_functions()

println("Loaded generated functions.")

tr = simulate(consts_compiled, (2,2,2,2,2));
println("Simulated a trace from the constant-inlined indexed model.")

# ### Circuit compilation
circuit = gen_fn_circuit(
    consts_compiled,
    map(d -> FiniteDomain(length(DiscreteIRTransforms.vals(d))), latent_domains()),
    Assess()
)
println("Circuit constructed.")

includet("../neurips_tracking/implementation_rules.jl")
println("Implemenation rules loaded.")

impl1 = implement(circuit, Spiking())
println("Circuit implemented once.")

impl_deep = Circuits.memoized_implement_deep(circuit, Spiking());
println("Circuit implemented deeply.")