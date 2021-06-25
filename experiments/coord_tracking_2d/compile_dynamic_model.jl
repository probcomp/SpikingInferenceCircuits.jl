using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

includet("dynamic_model.jl")

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 4

# a = GenFnWithInputDomains(initial_latent_model, ())
# b = SIC.replace_return_node(a)
# c = SIC.add_activator_input(b, :in)

# gf = SIC.DiscreteIRTransforms.add_activator_input(SIC.icpts(c.gf), c.activator_input_name)
# doms =    [
#         FiniteDomain(1), # activator input
#         (FiniteDomain((length ∘ DiscreteIRTransforms.vals)(x)) for x in c.gf.input_domains)...
#     ]

# circ = SIC.gen_fn_circuit(gf, doms, Assess())
smc = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(initial_proposal, obs_domains()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(step_proposal, latent_obs_domains()),
    [:xₜ, :vxₜ, :yₜ, :vyₜ],
    [:obsx, :obsy],
    [:xₜ, :vxₜ, :yₜ, :vyₜ],
    NPARTICLES()
)

println("SMC Circuit Constructed.")

includet("implementation_rules.jl")
println("Implementation rules loaded.")
impl = Circuits.memoized_implement_deep(smc, Spiking());
println("Circuit implemented deeply.")