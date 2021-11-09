using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using DynamicModels

includet("model.jl")
includet("inference.jl")

latent_domains()     = (xₜ=Positions(),)
obs_domains()         = (obs=Positions(),)
latent_obs_domains() = (latent_domains()..., obs_domains()...)

NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

includet("../utils/default_implementation_rules.jl")
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
    GenFnWithInputDomains(_exact_step_proposal, latent_obs_domains()),
    [:xₜ], [:obs], [:xₜ], NPARTICLES();
    truncation_minprob=MinProb()
)
println("SMC Circuit Constructed.")

impl = Circuits.memoized_implement_deep(smccircuit, Spiking());
#impl = Circuits.inline(impl)[1]
println("Circuit fully implemented using Poisson Process neurons.")

includet("../utils/simulation_utils.jl")

obs = [(obs = x,) for x in [5, 8, 10, 17, 17, 15, 11, 13, 10, 10, 12]]
inputs = get_smc_circuit_inputs(
                                RUNTIME(),
                                INTER_OBS_INTERVAL(),
                                obs
                               )
println("Constructed input spike sequence.")

events = simulate_and_get_events(impl, RUNTIME(), inputs; dir=@__DIR__)
println("Simulation completed!")

includet("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NPARTICLES(), NLATENTS())
