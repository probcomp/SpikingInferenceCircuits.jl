using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

includet("dynamic_model.jl")

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 4

smc = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(initial_proposal, obs_domains()),
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