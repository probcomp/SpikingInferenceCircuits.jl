using Revise
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using DynamicModels

includet("../experiments/velwalk1d/model.jl")
includet("../experiments/velwalk1d/pm_model.jl")
includet("../experiments/velwalk1d/inference.jl")
@load_generated_functions()

latent_domains()     = (xₜ=Positions(), vₜ=Vels())
obs_domains()         = (obs=Positions(),)

latent_obs_domains() = (latent_domains()..., obs_domains()...)
NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

includet("../experiments/utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), NVARS(), NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

smccircuit = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(_exact_init_proposal, obs_domains()),
    GenFnWithInputDomains(_approx_step_proposal, latent_obs_domains()),
    [:xₜ, :vₜ], [:obs], [:xₜ, :vₜ], NPARTICLES();
    truncation_minprob=MinProb()
)
println("SMC Circuit Constructed.")

impl = Circuits.memoized_implement_deep(smccircuit, Spiking());
