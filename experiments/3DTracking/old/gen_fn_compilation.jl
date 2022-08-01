using DiscreteIRTransforms

include("model.jl")

# latents = (moving_in_depthₜ, vₜ, heightₜ, xₜ, yₜ, rₜ)
latent_domains() = map(EnumeratedDomain, [
    Bools(), Vels(), Heights(), Xs(), Ys(), Rs()
])

obs_domains() = map(EnumeratedDomain, [Azimuths(), Altitudes()])
latent_obs_domains() = [latent_domains()..., obs_domains()...]

#########
# Model #
#########

### Compile IR to use simple conditional probability tables ###

function ir_compile_step_or_obs(model)
    lcpts = to_labeled_cpts(model, latent_domains())
    (cpts, dom_maps) = to_indexed_cpts(step, latent_domains())
    return (lcpts, cpts, dom_maps)
end

ir_compile_obs() = ir_compile_step_or_obs(obs_model)
ir_compile_step() = ir_compile_step_or_obs(step_model)

(obs_lcpts, obs_icpts) =  ir_compile_obs()
println("Performed IR compilation for obs model.")

(step_lcpts, step_icpts) =  ir_compile_step()
println("Performed IR compilation for step model.")

# TODO: do we want to inline constants?

@load_generated_functions()
simulate(obs_icpts, Tuple(1 for _ in latent_domains()))
println("Successfully simulated from indexed obs model.")

simulate(step_icpts, Tuple(1 for _ in latent_domains()))
println("Successfully simulated from indexed step model.")

### Compile simplified models into spiking neural networks ###

includet("../neurips_tracking/implementation_rules.jl")
println("Implemenation rules loaded.")

function compile_model_circuit(indexed_cpts_model, doms, model_name)
    circuit = gen_fn_circuit(
        indexed_cpts_model,
        map(d -> FiniteDomain(length(DiscreteIRTransforms.vals(d))), doms),
        Assess()
    )
    println("Constructed a Gen Fn cirucuit for $model_name.")

    impl_deep = Circuits.memoized_implement_deep(circuit, Spiking());
    println("Circuit implemented deeply.")
    
    return impl_deep
end

obs_model_spiking_network = compile_model_circuit(obs_icpts, latent_domains(), "obs")
step_model_spiking_network = compile_model_circuit(obs_icpts, latent_domains(), "step")