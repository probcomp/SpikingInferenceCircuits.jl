using SpikingInferenceCircuits
using Circuits, SpikingCircuits
include("compilable_model.jl")

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 4

rsmcstep = RecurrentSMCStep(
    SMCStep(
        GenFnWithInputDomains(step_model, latent_domains()),
        GenFnWithInputDomains(obs_model, latent_domains()),
        GenFnWithInputDomains(step_proposal, latent_obs_domains()),
        [:xₜ, :vxₜ, :yₜ, :vyₜ],
        [:obsx, :obsy],
        NPARTICLES()
    ),
    [:xₜ, :vxₜ, :yₜ, :vyₜ]
)
println("SMC Circuit Constructed.")

includet("implementation_rules.jl")
println("Implementation rules loaded.")
impl = Circuits.implement_deep(rsmcstep, Spiking())
println("Circuit implemented deeply.")